import pandas as pd
from functools import reduce

# File paths (update if needed)
ASSESSMENTS_PATH = "MathMatrix2425SY_AssessmentsItemized.csv"
GRADES_PATH = "MathMatrix2425SY_Grades.csv"
CONTENT_PATH = "MathMatrix2425SY_ContentProgress.csv"
EOC_PATH = "MathMatrix2425SY_EOCsurvey.csv"
TEACHER_PATH = "MathMatrix2425SY_TeacherBackgroundSurvey.csv"

OUTPUT_PATH = "merged_student_question_history.csv"


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV with UserId as string."""
    return pd.read_csv(path, dtype={"UserId": str})


def build_question_choices(assessments: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique question (across all attempts), collect the set of answer
    options as a single string.

    Output columns: QuesitonId, QuestionVersionId (if present), q_choices.
    """
    cols = [c for c in ["QuesitonId", "QuestionVersionId"] if c in assessments.columns]
    if not cols or "Answer" not in assessments.columns:
        return pd.DataFrame()

    tmp = (
        assessments.dropna(subset=cols)
        .groupby(cols, as_index=False)["Answer"]
        .agg(lambda s: [v for v in pd.unique(s.dropna().astype(str))])
    )

    tmp["q_choices"] = tmp["Answer"].apply(
        lambda vals: "[" + ", ".join(vals) + "]" if len(vals) else pd.NA
    )
    tmp = tmp.drop(columns=["Answer"])
    return tmp


def build_question_events(assessments: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (student, question attempt) with:
      - q_id: question id
      - q_correct: 1/0 (if available)
      - q_text: question text
      - q_choices: all options for that question
      - q_index: position in that student's sequence
    """
    # Choices from full table
    choices = build_question_choices(assessments)

    df = assessments.copy()

    # Keep only the selected option row per item (so one row = one attempt)
    if "AnswerChoiceSelected" in df.columns:
        df = df[df["AnswerChoiceSelected"] == 1].copy()

    # Correctness
    if "AnswerChoiceSelected_IsCorrect" in df.columns:
        df["q_correct"] = df["AnswerChoiceSelected_IsCorrect"]
    elif "CorrectResponse" in df.columns:
        df["q_correct"] = df["CorrectResponse"].astype(int)
    else:
        df["q_correct"] = pd.NA

    # Attach choices per question if possible
    if not choices.empty:
        on_cols = [c for c in ["QuesitonId", "QuestionVersionId"]
                   if c in df.columns and c in choices.columns]
        if on_cols:
            df = df.merge(choices, on=on_cols, how="left")
    else:
        df["q_choices"] = pd.NA

    # Sort within each student so q1, q2, q3... follow actual order
    sort_cols = [c for c in [
        "UserId",
        "TimeStarted",
        "TimeCompleted",
        "QuizTimeCompleted",
        "QuizId",
        "AttemptId",
        "QuestionNumber",
        "QuesitonId",
    ] if c in df.columns]
    df = df.sort_values(sort_cols)

    # Sequential index per student
    df["q_index"] = df.groupby("UserId").cumcount() + 1

    # Select and rename core columns
    col_map = {
        "QuesitonId": "q_id",
        "Question": "q_text",
        "q_correct": "q_correct",
        "q_choices": "q_choices",
    }
    keep = ["UserId", "q_index"] + [c for c in col_map.keys() if c in df.columns]
    df = df[keep].rename(columns=col_map)

    return df


def pivot_student_question_history(events: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format events into wide format:
      UserId,
      q1_id, q1_correct, q1_text, q1_choices,
      q2_id, q2_correct, ...
    """
    if events.empty:
        return pd.DataFrame(columns=["UserId"])

    wide = (
        events
        .set_index(["UserId", "q_index"])
        .unstack("q_index")
    )

    # Columns are MultiIndex: (field, q_index)
    new_cols = []
    for field, idx in wide.columns:
        if field == "q_id":
            suffix = "id"
        elif field == "q_correct":
            suffix = "correct"
        elif field == "q_text":
            suffix = "text"
        elif field == "q_choices":
            suffix = "choices"
        else:
            suffix = field
        new_cols.append(f"q{idx}_{suffix}")

    wide.columns = new_cols
    wide = wide.reset_index()

    return wide


def agg_categorical(series: pd.Series):
    """Pipe-join unique non-null string values."""
    vals = series.dropna().astype(str).unique()
    if len(vals) == 0:
        return pd.NA
    return " | ".join(vals)


def build_user_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Collapse any table to one row per UserId so we can safely merge.
    - Numeric cols -> mean
    - Non-numeric -> unique values joined with " | "
    Prefixed to avoid name clashes.
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

    rename_map = {
        col: f"{prefix}_{col}"
        for col in grouped.columns
        if col != "UserId"
    }
    grouped = grouped.rename(columns=rename_map)

    return grouped


def main():
    # Load
    assessments = load_csv(ASSESSMENTS_PATH)
    grades = load_csv(GRADES_PATH)
    content = load_csv(CONTENT_PATH)
    eoc = load_csv(EOC_PATH)
    teacher = load_csv(TEACHER_PATH)

    # Core: per-student question-answering history
    events = build_question_events(assessments)
    question_history = pivot_student_question_history(events)

    # Side tables: per-student aggregates from other datasets
    grades_user = build_user_features(grades, "grades")
    content_user = build_user_features(content, "content")
    eoc_user = build_user_features(eoc, "eoc")
    teacher_user = build_user_features(teacher, "teacher")

    # Merge all by UserId, keeping students with assessment data
    dfs = [question_history, grades_user, content_user, eoc_user, teacher_user]
    dfs = [df for df in dfs if not df.empty]
    merged = reduce(lambda left, right: left.merge(right, on="UserId", how="left"), dfs)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged file to {OUTPUT_PATH} with shape {merged.shape}")


if __name__ == "__main__":
    main()
