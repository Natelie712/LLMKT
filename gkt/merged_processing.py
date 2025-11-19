
import os
import re
import numpy as np
import pandas as pd

class MergedGKTProcessor:
    """
    Convert wide merged dataset (qN_id, qN_answerchoiceselected_iscorrect, etc.)
    into long-format CSV suitable for GKT training.

    Output format columns:
        user_id
        skill_id   (concept = question id)
        problem_id (same as skill_id, but GKT code expects both)
        correct    (0/1)
    """

    def __init__(self,
                 source_csv="merged_student_question_history.csv",
                 output_csv="data/merged_gkt.csv",
                 min_seq_len=3):
        self.source_csv = source_csv
        self.output_csv = output_csv
        self.min_seq_len = min_seq_len
        os.makedirs("data", exist_ok=True)

    @staticmethod
    def _clean_id(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        try:
            fv = float(val)
            if fv.is_integer():
                return int(fv)
            return val
        except:
            return val

    @staticmethod
    def _to_binary(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        if isinstance(val, bool):
            return 1 if val else 0
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"1", "true", "t", "yes"}:
                return 1
            if v in {"0", "false", "f", "no"}:
                return 0
        try:
            fv = float(val)
            if fv == 1.0:
                return 1
            if fv == 0.0:
                return 0
        except:
            return None
        return None

    def _detect_qgroups(self, df):
        patt = re.compile(r"^q(\d+)_(.+)$")
        groups = {}
        for col in df.columns:
            m = patt.match(col)
            if not m:
                continue
            idx = int(m.group(1))
            suf = m.group(2)
            if idx not in groups:
                groups[idx] = {}
            groups[idx][suf] = col
        return groups

    def process(self):
        print("Loading", self.source_csv)
        df = pd.read_csv(self.source_csv, low_memory=False)

        # unify user id
        if "user_id" not in df.columns and "UserId" in df.columns:
            df = df.rename(columns={"UserId": "user_id"})
        if "user_id" not in df.columns:
            raise ValueError("Expected 'user_id' or 'UserId' in dataset.")

        groups = self._detect_qgroups(df)
        q_indices = sorted(groups.keys())
        if q_indices:
            print(f"Detecting question indices from qN_* columns ...")
            print(f"Found {len(q_indices)} question slots: from q{q_indices[0]} to q{q_indices[-1]}")

        rows = []
        for _, row in df.iterrows():
            user = row["user_id"]
            seq = []
            for qn in q_indices:
                g = groups[qn]

                # question id
                qid_col = g.get("id")
                if not qid_col:
                    continue
                qid = self._clean_id(row[qid_col])
                if qid is None:
                    continue

                # correctness priority
                corr = None
                for suf in ["correct", "answerchoiceselected_iscorrect"]:
                    if suf in g:
                        corr = self._to_binary(row[g[suf]])
                        if corr is not None:
                            break
                if corr is None:
                    continue

                seq.append((qid, corr))

            if len(seq) >= self.min_seq_len:
                for qid, corr in seq:
                    rows.append({
                        "user_id": user,
                        "skill_id": qid,
                        "problem_id": qid,
                        "correct": corr
                    })

        out_df = pd.DataFrame(rows)

        # Report unique questions for embedding vocabulary size
        if not out_df.empty:
            unique_q = out_df["skill_id"].nunique()
            print(f"Number of unique questions: {unique_q}")
        print("Saving to", self.output_csv)
        out_df.to_csv(self.output_csv, index=False)
        print("Done. Total rows:", len(out_df))


if __name__ == "__main__":
    proc = MergedGKTProcessor()
    proc.process()
