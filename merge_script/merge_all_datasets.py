import pandas as pd
from functools import reduce
import re

# Input file paths
ASSESSMENTS_PATH = "MathMatrix2425SY_AssessmentsItemized.csv"
GRADES_PATH = "MathMatrix2425SY_Grades.csv"
CONTENT_PATH = "MathMatrix2425SY_ContentProgress.csv"
EOC_PATH = "MathMatrix2425SY_EOCsurvey.csv"
TEACHER_PATH = "MathMatrix2425SY_TeacherBackgroundSurvey.csv"

OUTPUT_PATH = "merged_student_question_history.csv"


def load_csv(path: str) -> pd.DataFrame:
    """Robust CSV loader."""
    return pd.read_csv(path, low_memory=False)


def safe_suffix(name: str) -> str:
    """
    Convert a column name into a safe suffix:
    - lowercase
    - non-alphanumeric -> underscore
    - strip leading/trailing underscores
    """
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return s or "col"


def build_assessment_events(assessments: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (UserId, question attempt) from the AssessmentsItemized dataset.

    We:
    - Keep all columns that may have educational meaning.
    - Create q_correct from available correctness columns.
    - Sort within each user to get a stable question order.
    - Assign q_index (1,2,3,...) per student.
    """
    df = assessments.copy()

    if "UserId" not in df.columns:
        raise ValueError("Assessments file missing 'UserId'.")

    # If present, restrict to selected choices (one row per answered item)
    if "AnswerChoiceSelected" in df.columns:
        df = df[df["AnswerChoiceSelected"] == 1].copy()

    # Derive correctness label
    if "AnswerChoiceSelected_IsCorrect" in df.columns:
        df["q_correct"] = df["AnswerChoiceSelected_IsCorrect"].astype("float32")
    elif "CorrectResponse" in df.columns:
        # Fallback: if CorrectResponse is coded (e.g., 0/1 or True/False-like)
        try:
            df["q_correct"] = pd.to_numeric(df["CorrectResponse"], errors="coerce")
        except Exception:
            df["q_correct"] = pd.NA
    else:
        df["q_correct"] = pd.NA

    # Sort columns to approximate true temporal / logical order
    sort_cols = [c for c in [
        "UserId",
        "QuizTimeCompleted",
        "TimeCompleted",
        "TimeStarted",
        "AttemptId",
        "AttemptNumber",
        "QuizId",
        "QuestionNumber",
        "QuesitonId",
    ] if c in df.columns]

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Assign sequence index per student
    df["q_index"] = df.groupby("UserId").cumcount() + 1

    # Keep all columns that may be meaningful.
    # We only drop obviously technical artifacts if any show up;
    # here we keep everything by default.
    # Later, pivot_question_history will handle naming.
    keep_cols = [c for c in df.columns if c not in []]
    events = df[keep_cols].copy()

    return events


def pivot_question_history(events: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format assessment events into wide per-student format.

    For each q_index k and base column X, we create:
        q{k}_id         (from QuesitonId)
        q{k}_correct    (from q_correct)
        q{k}_<suffix>   (for all other columns)

    This preserves:
    - the per-student sequence structure
    - as many assessment-level columns as possible
    - compatibility with existing preprocess_dataset.py (expects q*_id, q*_correct).
    """
    if events.empty:
        return pd.DataFrame(columns=["UserId"])

    if "UserId" not in events.columns or "q_index" not in events.columns:
        raise ValueError("Events must contain 'UserId' and 'q_index'.")

    events = events.set_index(["UserId", "q_index"])
    base_cols = [c for c in events.columns if c not in ["UserId", "q_index"]]

    wide = events[base_cols].unstack("q_index")

    new_cols = []
    for base, idx in wide.columns:
        # Special handling for DKT compatibility
        if base == "QuesitonId":
            col_name = f"q{idx}_id"
        elif base == "q_correct":
            col_name = f"q{idx}_correct"
        else:
            suf = safe_suffix(base)
            col_name = f"q{idx}_{suf}"
        new_cols.append(col_name)

    wide.columns = new_cols
    wide = wide.reset_index()

    return wide


def agg_categorical(series: pd.Series) -> str:
    """Join unique non-null string values. Preserves information without losing categories."""
    vals = series.dropna().astype(str).unique()
    if len(vals) == 0:
        return pd.NA
    return " | ".join(vals)


def build_user_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Generic user-level feature builder for:
        - ContentProgress
        - Grades
        - EOCsurvey
        - TeacherBackgroundSurvey

    Strategy:
    - Group by UserId.
    - For each other column:
        - numeric -> mean
        - non-numeric -> unique values joined with " | "
    - Prefix all columns with dataset-specific prefix (e.g., 'content_', 'grades_').
    - This preserves as many columns as possible in an aggregated, future-proof way.
    """
    if "UserId" not in df.columns:
        return pd.DataFrame(columns=["UserId"])

    df = df.copy()
    agg_spec = {}

    for col in df.columns:
        if col == "UserId":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_spec[col] = "mean"
        else:
            agg_spec[col] = agg_categorical

    if not agg_spec:
        return df[["UserId"]].drop_duplicates()

    grouped = df.groupby("UserId", as_index=False).agg(agg_spec)

    # Prefix all non-UserId columns
    rename_map = {}
    for col in grouped.columns:
        if col == "UserId":
            continue
        rename_map[col] = f"{prefix}_{safe_suffix(col)}"

    grouped = grouped.rename(columns=rename_map)
    return grouped


def main():
    # Load all datasets
    assessments = load_csv(ASSESSMENTS_PATH)
    grades = load_csv(GRADES_PATH)
    content = load_csv(CONTENT_PATH)
    eoc = load_csv(EOC_PATH)
    teacher = load_csv(TEACHER_PATH)

    # 1) Build per-question events from AssessmentsItemized
    events = build_assessment_events(assessments)

    # 2) Pivot into wide per-student question history (q1_*, q2_*, ...)
    question_history = pivot_question_history(events)

    # 3) Build user-level aggregates from the other datasets
    grades_user = build_user_features(grades, "grades")
    content_user = build_user_features(content, "content")
    eoc_user = build_user_features(eoc, "eoc")
    teacher_user = build_user_features(teacher, "teacher")

    # 4) Merge everything on UserId
    dfs = [question_history, grades_user, content_user, eoc_user, teacher_user]
    dfs = [df for df in dfs if not df.empty]

    if not dfs:
        raise RuntimeError("No data to merge; please check input files.")

    merged = reduce(
        lambda left, right: left.merge(right, on="UserId", how="left"),
        dfs
    )

    # 5) Save final merged dataset
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged file to {OUTPUT_PATH} with shape {merged.shape}")


if __name__ == "__main__":
    main()
