
# dkt_preprocess_merged.py
# Preprocess merged_student_question_history.csv into DKT-ready sequences.
#
# Usage (from repo root):
#   python dkt_preprocess_merged.py \
#       --input merged_student_question_history.csv \
#       --output_dir dkt_processed \
#       --min_seq_len 3 \
#       --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1
#
# IMPORTANT - CSV Structure (as of latest version):
#   - Assessment questions: 305 columns (q1_id to q305_id with q*_correct columns)
#     * Multi-select questions now count as ONE question (not multiple)
#     * Correctness requires ALL correct choices AND NO incorrect choices
#   - Grade columns: Pivoted format (grades_<item_name> per grade item)
#   - Content columns: Topic-specific (content_<topic>_totaltime, etc.)
#   - EOC Survey columns: Per-question format (eoc_<survey>_q<N>_question/answer)
#
# This script:
#   1. Reads the wide merged CSV where each row is one student / teacher.
#   2. Dynamically detects question columns (q*_id) - handles any count.
#   3. Extracts ordered question and correctness sequences from qN_* blocks.
#   4. Calculates GLOBAL completion rates (questions_answered / TOTAL_Q_SLOTS).
#   5. Maps raw question ids to contiguous indices [0, num_questions - 1].
#   6. Drops users with short sequences (length < min_seq_len).
#   7. Splits users into train / valid / test at the user level.
#   8. Saves the result as pickles plus a small JSON metadata file.

import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess merged CSV for DKT")
    parser.add_argument(
        "--input",
        type=str,
        default="merged_student_question_history.csv",
        help="Path to merged wide CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dkt_processed",
        help="Directory to save processed outputs",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=3,
        help="Minimum sequence length per user",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (user level)",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (user level)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split ratio (user level)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    return parser.parse_args()


def find_question_indices(columns: List[str]) -> List[int]:
    """Return sorted list of N such that qN_id exists."""
    q_indices = []
    for col in columns:
        if col.startswith("q") and col.endswith("_id"):
            # col is like "q123_id"
            try:
                middle = col[1 : col.rindex("_")]
                idx = int(middle)
                q_indices.append(idx)
            except Exception:
                continue
    q_indices = sorted(set(q_indices))
    if not q_indices:
        raise ValueError("No qN_id columns found in input CSV.")
    return q_indices


def normalize_correct_value(val) -> int:
    """Normalize correctness value to 0 or 1.

    Priority:
      - If val is NaN or None: raise ValueError.
      - If numeric: > 0 -> 1 else 0.
      - If string: map common truthy/falsey tokens.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        raise ValueError("Missing correctness")

    # numeric case
    if isinstance(val, (int, float, np.integer, np.floating)):
        return 1 if float(val) > 0.0 else 0

    s = str(val).strip().lower()
    if s == "":
        raise ValueError("Empty correctness")

    truthy = {"1", "true", "t", "yes", "y", "correct"}
    falsey = {"0", "false", "f", "no", "n", "incorrect"}

    if s in truthy:
        return 1
    if s in falsey:
        return 0

    # Fall back: try to parse as float
    try:
        f = float(s)
        return 1 if f > 0.0 else 0
    except Exception:
        # If totally unknown, raise and let caller handle skipping this step.
        raise ValueError(f"Unrecognized correctness value: {val!r}")


def extract_sequence_for_row(
    row: pd.Series,
    q_indices: List[int],
    correctness_priority: List[str],
) -> Tuple[List[str], List[int]]:
    """Extract (question_ids, correctness) sequence for a single row.

    correctness_priority is a list of suffixes like:
      ["answerchoiceselected_iscorrect", "correctresponse", "correct"]
    """
    q_ids: List[str] = []
    r_vals: List[int] = []

    for q_idx in q_indices:
        base = f"q{q_idx}"
        id_col = f"{base}_id"

        if id_col not in row or pd.isna(row[id_col]):
            # No question at this position
            continue

        raw_qid = row[id_col]
        qid_str = str(raw_qid).strip()
        if qid_str == "" or qid_str.lower() == "nan":
            continue

        # Find correctness using priority list
        correct_val = None
        for suffix in correctness_priority:
            col_name = f"{base}_{suffix}"
            if col_name in row and not pd.isna(row[col_name]):
                correct_val = row[col_name]
                break

        if correct_val is None:
            # No correctness information, skip this position
            continue

        try:
            norm_correct = normalize_correct_value(correct_val)
        except ValueError:
            # Could not interpret this correctness, skip this position
            continue

        q_ids.append(qid_str)
        r_vals.append(norm_correct)

    return q_ids, r_vals


def map_questions_to_indices(all_q_ids: List[str]) -> Dict[str, int]:
    """Map unique question ids to 0..num_questions-1."""
    uniq = sorted(set(all_q_ids))
    return {qid: i for i, qid in enumerate(uniq)}


def split_indices(n: int, train_ratio: float, valid_ratio: float, test_ratio: float, rng: np.random.Generator):
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/valid/test ratios must sum to 1.")

    indices = np.arange(n)
    rng.shuffle(indices)

    train_end = int(train_ratio * n)
    valid_end = train_end + int(valid_ratio * n)

    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]

    return train_idx, valid_idx, test_idx


def save_split(
    path: str,
    users: List[str],
    q_seqs: List[List[int]],
    r_seqs: List[List[int]],
    seq_lens: List[int],
    completion_rates: List[float],
):
    data = {
        "users": users,
        "q_seqs": q_seqs,
        "r_seqs": r_seqs,
        "seq_lengths": seq_lens,
        "completion_rates": completion_rates,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading input CSV from {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    if "UserId" not in df.columns:
        raise ValueError("Expected a 'UserId' column in the input CSV.")

    print("Detecting question indices from qN_id columns ...")
    q_indices = find_question_indices(list(df.columns))
    print(f"Found {len(q_indices)} question slots: from q{q_indices[0]} to q{q_indices[-1]}")

    # TOTAL_Q_SLOTS is the GLOBAL constant for completion rate calculation
    # This represents the maximum possible questions any student could answer
    # Current CSV structure: 305 questions (multi-select now counts as 1)
    # Used for fairness metrics: completion_rate = student_questions / TOTAL_Q_SLOTS
    TOTAL_Q_SLOTS = len(q_indices)

    # Define correctness priority for each qN_*
    correctness_priority = [
        "correct",
        "answerchoiceselected_iscorrect",
    ]

    all_user_ids: List[str] = []
    all_qid_strings: List[List[str]] = []
    all_correctness: List[List[int]] = []

    print("Extracting sequences per user (this may take a moment) ...")
    for idx, row in df.iterrows():
        user_id = str(row["UserId"]).strip()
        if user_id == "" or user_id.lower() == "nan":
            continue

        q_ids, r_vals = extract_sequence_for_row(row, q_indices, correctness_priority)

        if len(q_ids) < args.min_seq_len:
            continue

        all_user_ids.append(user_id)
        all_qid_strings.append(q_ids)
        all_correctness.append(r_vals)

        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1} rows ...")

    num_users = len(all_user_ids)
    if num_users == 0:
        raise ValueError("No users with valid sequences found. Check preprocessing assumptions.")

    print(f"Kept {num_users} users with seq_len >= {args.min_seq_len}.")

    # Build global question-id to index mapping
    print("Building question id to index mapping ...")
    flat_qids = [qid for seq in all_qid_strings for qid in seq]
    qid2idx = map_questions_to_indices(flat_qids)
    num_questions = len(qid2idx)
    print(f"Number of unique questions: {num_questions}")

    # Map question ids in sequences to indices
    q_seqs_idx: List[List[int]] = []
    r_seqs_idx: List[List[int]] = []
    seq_lengths: List[int] = []
    completion_rates: List[float] = []

    # CRITICAL: Completion rate is a STABLE property calculated ONCE per student
    # Formula: completion_rate = (total questions answered by student) / TOTAL_Q_SLOTS
    # This ensures the same student ALWAYS maps to the same fairness bin
    # regardless of batch composition or sequence chunking
    for q_ids, r_vals in zip(all_qid_strings, all_correctness):
        q_idx_seq = [qid2idx[qid] for qid in q_ids]
        q_seqs_idx.append(q_idx_seq)
        r_seqs_idx.append(list(r_vals))
        L = len(q_idx_seq)  # Total questions answered by THIS student
        seq_lengths.append(L)
        # Global completion rate (stable across all batches)
        completion_rates.append(L / float(TOTAL_Q_SLOTS))

    # Train / valid / test split at user level
    rng = np.random.default_rng(args.seed)
    train_idx, valid_idx, test_idx = split_indices(
        n=num_users,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        rng=rng,
    )

    def subset(indices: np.ndarray):
        return (
            [all_user_ids[i] for i in indices],
            [q_seqs_idx[i] for i in indices],
            [r_seqs_idx[i] for i in indices],
            [seq_lengths[i] for i in indices],
            [completion_rates[i] for i in indices],
        )

    train_users, train_q, train_r, train_len, train_comp = subset(train_idx)
    valid_users, valid_q, valid_r, valid_len, valid_comp = subset(valid_idx)
    test_users, test_q, test_r, test_len, test_comp = subset(test_idx)

    # Save splits
    train_path = os.path.join(args.output_dir, "dkt_train.pkl")
    valid_path = os.path.join(args.output_dir, "dkt_valid.pkl")
    test_path = os.path.join(args.output_dir, "dkt_test.pkl")

    print(f"Saving train split to {train_path} ...")
    save_split(train_path, train_users, train_q, train_r, train_len, train_comp)
    print(f"Saving valid split to {valid_path} ...")
    save_split(valid_path, valid_users, valid_q, valid_r, valid_len, valid_comp)
    print(f"Saving test split to {test_path} ...")
    save_split(test_path, test_users, test_q, test_r, test_len, test_comp)

    # Save metadata
    meta = {
        "num_questions": num_questions,
        "total_q_slots": TOTAL_Q_SLOTS,  # Global constant for completion rates (305 as of latest CSV)
        "min_seq_len": args.min_seq_len,
        "train_users": len(train_users),
        "valid_users": len(valid_users),
        "test_users": len(test_users),
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    meta_path = os.path.join(args.output_dir, "dkt_metadata.json")
    print(f"Saving metadata to {meta_path} ...")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save also the mapping from question index to original id for reference
    idx2qid = {idx: qid for qid, idx in qid2idx.items()}
    mapping_path = os.path.join(args.output_dir, "dkt_question_mapping.json")
    print(f"Saving question id mapping to {mapping_path} ...")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(idx2qid, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
