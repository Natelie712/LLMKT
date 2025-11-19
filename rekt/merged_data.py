
import os
import re
import numpy as np
import pandas as pd

from merged_utils import (
    count,
    save_list,
    save_dict,
    write_lists,
)


class MergedDataProcess:
    """
    Preprocess `merged_student_question_history.csv` into ReKT text format.

    Assumptions
    ----------
    - Each row = one unique student/teacher.
    - Student id column is `UserId` (will be renamed to `user_id`).
    - Question attempts are stored in wide form as groups of columns:

        q1_id
        q1_answerchoiceselected_iscorrect
        q1_correctresponse
        q1_correct
        q2_id
        q2_answerchoiceselected_iscorrect
        ...

      where:

        - Question identity = qN_id
        - Concept/skill id = question id (1-to-1 mapping)
        - Correctness priority per attempt:

            1) qN_correct
            2) qN_answerchoiceselected_iscorrect

          Any value that can be interpreted as 0/1 is accepted:
          0/1, '0'/'1', True/False, 'True'/'False', etc.

    Outputs
    -------
    In `save_folder` (default: data/MERGED):

      - train_question.txt
      - test_question.txt
      - train_skill.txt
      - test_skill.txt
      - ques_skill.csv

    Each *question/skill* file uses the standard ReKT format:

        line 1: sequence length L
        line 2: comma-separated question (or skill) ids
        line 3: comma-separated 0/1 answers

    You should call this once before training and then point
    `mp2path['merged']` in main.py to these files.
    """

    def __init__(
        self,
        data_path: str = "merged_student_question_history.csv",
        save_folder: str = "data/MERGED",
        train_user_ratio: float = 0.8,
        min_seq_len: int = 3,
        random_state: int = 2024,
    ):
        self.data_path = data_path
        self.save_folder = save_folder
        self.train_user_ratio = train_user_ratio
        self.min_seq_len = min_seq_len
        self.random_state = random_state

        os.makedirs(self.save_folder, exist_ok=True)

        # optional subdirs if you want to mirror ASSIST09 layout
        self.encode_folder = os.path.join(self.save_folder, "encode")
        os.makedirs(self.encode_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_binary(value):
        """Convert various truthy/falsey representations to 0/1 or None."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None

        # strings
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "nan"}:
                return None
            if v in {"1", "true", "t", "yes", "y"}:
                return 1
            if v in {"0", "false", "f", "no", "n"}:
                return 0
            # try numeric
            try:
                num = float(v)
                if num == 1.0:
                    return 1
                if num == 0.0:
                    return 0
            except Exception:
                return None

        # booleans
        if isinstance(value, bool):
            return 1 if value else 0

        # numeric
        try:
            num = float(value)
            if num == 1.0:
                return 1
            if num == 0.0:
                return 0
        except Exception:
            return None

        return None

    @staticmethod
    def _clean_qid(value):
        """Normalize question id to an int if possible, else string."""
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        # try integer-ish
        try:
            num = float(value)
            if num.is_integer():
                return int(num)
            return value
        except Exception:
            return value

    def _identify_q_columns(self, df: pd.DataFrame):
        """
        Inspect columns and return:

            q_indices: sorted list of N where we saw columns like 'qN_*'
            meta: dict[N] = {
                "id_col":  'qN_id' if present, else None,
                "corr_cols": {suffix: colname, ...}
            }
        """
        pattern = re.compile(r"^q(\d+)_([A-Za-z0-9_]+)$")

        q_indices = set()
        meta = {}

        for col in df.columns:
            m = pattern.match(col)
            if not m:
                continue
            N = int(m.group(1))
            suffix = m.group(2)

            q_indices.add(N)
            if N not in meta:
                meta[N] = {"id_col": None, "corr_cols": {}}

            if suffix == "id":
                meta[N]["id_col"] = col
            # we still record *all* suffixes; we will only *use*
            # the correctness-related ones later.
            meta[N]["corr_cols"][suffix] = col

        q_indices = sorted(q_indices)
        return q_indices, meta

    def _extract_sequences(self, df: pd.DataFrame):
        """
        Wide -> per-user sequences.

        Returns
        -------
        user_ids : list
        ques_seqs : list[list[raw_qid]]
        ans_seqs : list[list[0/1]]
        """
        # Identify qN_* structure once
        q_indices, meta = self._identify_q_columns(df)

        user_ids = []
        ques_seqs = []
        ans_seqs = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            user_id = row_dict.get("user_id")
            q_seq = []
            a_seq = []

            for N in q_indices:
                info = meta[N]
                id_col = info.get("id_col")

                if not id_col or id_col not in row_dict:
                    continue

                raw_qid = self._clean_qid(row_dict.get(id_col))
                if raw_qid is None:
                    continue

                # correctness priority: correct > answerchoiceselected_iscorrect
                corr_val = None
                corr_suffix_order = [
                    "correct",
                    "answerchoiceselected_iscorrect",
                ]
                for suffix in corr_suffix_order:
                    colname = info["corr_cols"].get(suffix)
                    if not colname:
                        continue
                    v = row_dict.get(colname)
                    b = self._to_binary(v)
                    if b is not None:
                        corr_val = b
                        break

                if corr_val is None:
                    # cannot determine correctness, skip this attempt
                    continue

                q_seq.append(raw_qid)
                a_seq.append(int(corr_val))

            if len(q_seq) >= self.min_seq_len:
                user_ids.append(user_id)
                ques_seqs.append(q_seq)
                ans_seqs.append(a_seq)

        return user_ids, ques_seqs, ans_seqs

    def _encode_questions_and_skills(self, ques_seqs):
        """
        Map raw question ids to contiguous indices and define
        concepts = questions (1-to-1).
        """
        all_qids = set()
        for seq in ques_seqs:
            all_qids.update(seq)

        all_qids = sorted(list(all_qids), key=lambda x: (isinstance(x, str), x))
        qid2idx = {qid: idx for idx, qid in enumerate(all_qids)}

        # concepts == questions
        skill2idx = {qid: idx for qid, idx in qid2idx.items()}

        # save dictionaries for reference / reuse
        save_dict(qid2idx, os.path.join(self.encode_folder, "question_id_dict.txt"))
        save_dict(skill2idx, os.path.join(self.encode_folder, "skill_id_dict.txt"))

        return qid2idx, skill2idx

    def _train_valid_test_split(self, user_ids, train_ratio=0.8, valid_ratio=0.1):
        """Split user indices into train/valid/test by user.

        Ensures disjoint sets and uses the class random_state for reproducibility.
        """
        rng = np.random.RandomState(self.random_state)
        n_users = len(user_ids)
        indices = np.arange(n_users)
        rng.shuffle(indices)

        n_train = int(n_users * train_ratio)
        n_valid = int(n_users * valid_ratio)
        n_test = n_users - n_train - n_valid

        train_idx = indices[:n_train]
        valid_idx = indices[n_train:n_train + n_valid]
        test_idx = indices[n_train + n_valid:]

        # Safety clamp in case of small datasets
        if n_test < 0:
            test_idx = np.array([], dtype=int)
        return train_idx, valid_idx, test_idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self):
        """
        Run full preprocessing pipeline and write ReKT files.
        """
        print("=== Loading merged data from:", self.data_path)
        df = pd.read_csv(self.data_path, low_memory=False)

        # unify user id column name
        if "user_id" not in df.columns and "UserId" in df.columns:
            df = df.rename(columns={"UserId": "user_id"})

        if "user_id" not in df.columns:
            raise ValueError("Could not find 'user_id' or 'UserId' column in merged dataset.")

        print("Total rows (students):", len(df))

        # Wide -> sequences
        user_ids, ques_seqs, ans_seqs = self._extract_sequences(df)

        print("Users with valid sequences (len >= %d): %d" % (self.min_seq_len, len(user_ids)))

        # Long-format view for basic stats
        rows = []
        for uid, qs, ans in zip(user_ids, ques_seqs, ans_seqs):
            for q, a in zip(qs, ans):
                rows.append({"user_id": uid, "problem_id": q, "skill_id": q, "correct": a})
        long_df = pd.DataFrame(rows)
        count("Merged (long) summary", long_df)

        # Encode question / skill ids
        qid2idx, skill2idx = self._encode_questions_and_skills(ques_seqs)

        # Map sequences to encoded indices
        encoded_q_seqs = [[qid2idx[q] for q in qs] for qs in ques_seqs]
        encoded_s_seqs = [[skill2idx[q] for q in qs] for qs in ques_seqs]

        # Train / valid / test split at user level (80/10/10)
        train_idx, valid_idx, test_idx = self._train_valid_test_split(user_ids, 0.8, 0.1)

        def _subset(indices):
            seq_lens = []
            qs_list = []
            ans_list = []
            skills_list = []

            for i in indices:
                qs = encoded_q_seqs[i]
                ss = encoded_s_seqs[i]
                aa = ans_seqs[i]

                assert len(qs) == len(ss) == len(aa)
                L = len(qs)

                seq_lens.append(L)
                qs_list.append(qs)
                skills_list.append(ss)
                ans_list.append(aa)

            return seq_lens, qs_list, skills_list, ans_list

        train_seq_lens, train_qs, train_skills, train_ans = _subset(train_idx)
        valid_seq_lens, valid_qs, valid_skills, valid_ans = _subset(valid_idx)
        test_seq_lens, test_qs, test_skills, test_ans = _subset(test_idx)

        # Write ReKT text files
        train_question_path = os.path.join(self.save_folder, "train_question.txt")
        valid_question_path = os.path.join(self.save_folder, "valid_question.txt")
        test_question_path = os.path.join(self.save_folder, "test_question.txt")
        train_skill_path = os.path.join(self.save_folder, "train_skill.txt")
        valid_skill_path = os.path.join(self.save_folder, "valid_skill.txt")
        test_skill_path = os.path.join(self.save_folder, "test_skill.txt")

        write_lists(train_seq_lens, train_qs, train_ans, train_question_path)
        write_lists(valid_seq_lens, valid_qs, valid_ans, valid_question_path)
        write_lists(test_seq_lens, test_qs, test_ans, test_question_path)
        write_lists(train_seq_lens, train_skills, train_ans, train_skill_path)
        write_lists(valid_seq_lens, valid_skills, valid_ans, valid_skill_path)
        write_lists(test_seq_lens, test_skills, test_ans, test_skill_path)

        print("Wrote:")
        print("  ", train_question_path)
        print("  ", valid_question_path)
        print("  ", test_question_path)
        print("  ", train_skill_path)
        print("  ", valid_skill_path)
        print("  ", test_skill_path)

        # ques_skill.csv: question_idx, skill_idx (1-to-1 mapping)
        qs_pairs = [[idx, skill2idx[qid]] for qid, idx in qid2idx.items()]
        qs_df = pd.DataFrame(qs_pairs, columns=["question_idx", "skill_idx"])
        ques_skill_path = os.path.join(self.save_folder, "ques_skill.csv")
        qs_df.to_csv(ques_skill_path, index=False)
        print("  ", ques_skill_path)

        print("Finished preprocessing merged dataset.")


if __name__ == "__main__":
    processor = MergedDataProcess()
    processor.process()
