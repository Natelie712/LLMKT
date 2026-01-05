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
    Build one row per (UserId, UNIQUE QUESTION) from the AssessmentsItemized dataset.
    
    Key changes from original:
    - Groups by (UserId, QuesitonId, AttemptId) to handle multi-select questions
    - For each question attempt, aggregates correctness across all selected choices
    - Question is marked correct only if ALL selected choices are correct
    - Preserves question text for LBKT compatibility
    
    This ensures:
    - Multi-select questions count as ONE question attempt
    - Correctness reflects overall question correctness, not per-choice
    - Student engagement measured by unique questions, not answer choices
    """
    df = assessments.copy()

    if "UserId" not in df.columns:
        raise ValueError("Assessments file missing 'UserId'.")

    # Ensure we have required columns
    if "QuesitonId" not in df.columns:
        raise ValueError("Assessments file missing 'QuesitonId'.")
    
    # Strategy: Check if student's answer matches the correct answer for the question
    # 1. For each question attempt, find all choices where CorrectResponse=TRUE
    # 2. Check if student selected exactly those choices (no more, no less)
    
    # Group by question attempt BEFORE filtering
    group_cols = ["UserId", "QuesitonId"]
    if "AttemptId" in df.columns:
        group_cols.append("AttemptId")
    elif "TimeCompleted" in df.columns:
        group_cols.append("TimeCompleted")
    
    # For each question attempt, check correctness
    def compute_question_correctness(group):
        """
        Determine if question answered correctly by checking:
        - All CorrectResponse=TRUE choices are selected (no missing correct answers)
        - No CorrectResponse=FALSE choices are selected (no wrong answers selected)
        """
        # Parse CorrectResponse as boolean if it exists
        if "CorrectResponse" in group.columns:
            correct_choices = group["CorrectResponse"].fillna(False)
            # Handle various representations of True/False
            if correct_choices.dtype == 'object':
                correct_choices = correct_choices.astype(str).str.upper().isin(['TRUE', '1', '1.0', 'T', 'YES'])
            else:
                correct_choices = correct_choices.astype(bool)
        else:
            # Fallback: assume single-choice question, use AnswerChoiceSelected_IsCorrect
            correct_choices = pd.Series([False] * len(group), index=group.index)
        
        # Which choices did student select?
        selected = group["AnswerChoiceSelected"].fillna(0) == 1
        
        # Question is correct if:
        # - All correct choices are selected: (correct_choices & selected).sum() == correct_choices.sum()
        # - No incorrect choices are selected: (~correct_choices & selected).sum() == 0
        num_correct = correct_choices.sum()
        num_correct_selected = (correct_choices & selected).sum()
        num_incorrect_selected = (~correct_choices & selected).sum()
        
        is_correct = (num_correct_selected == num_correct) and (num_incorrect_selected == 0)
        
        return pd.Series({
            "q_correct": 1.0 if is_correct else 0.0
        })
    
    # Apply to each question attempt
    correctness = df.groupby(group_cols, as_index=False).apply(compute_question_correctness)
    
    # Now filter to selected choices only for extracting other metadata
    df_selected = df[df["AnswerChoiceSelected"] == 1].copy() if "AnswerChoiceSelected" in df.columns else df.copy()
    
    # For each question attempt, keep first row's metadata
    agg_dict = {}
    
    # Add other columns we want to preserve (take first value from selected choices)
    preserve_cols = [
        "QuizTimeCompleted", "TimeCompleted", "TimeStarted",
        "AttemptId", "AttemptNumber", "QuizId", "QuestionNumber", "QuesitonId"
    ]
    
    for col in preserve_cols:
        if col in df_selected.columns and col not in group_cols:
            agg_dict[col] = "first"
    
    # Aggregate metadata from selected choices
    if agg_dict:
        metadata = df_selected.groupby(group_cols, as_index=False).agg(agg_dict)
    else:
        metadata = df_selected[group_cols].drop_duplicates()
    
    # Merge correctness with metadata
    question_level = correctness.merge(metadata, on=group_cols, how="left")
    
    # Sort by user and time to get chronological order
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
    ] if c in question_level.columns]
    
    if sort_cols:
        question_level = question_level.sort_values(sort_cols).reset_index(drop=True)
    
    # Assign sequence index per student (now counting unique questions, not choices)
    question_level["q_index"] = question_level.groupby("UserId").cumcount() + 1
    
    return question_level


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
    - compatibility with existing preprocess_dataset.py (expects q*_id, q*_correct)
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


def build_grades_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build grade features by pivoting grade items into columns.
    
    Each user has multiple grade items (different assignments/categories).
    We pivot so each grade item becomes its own column with the grade value.
    
    Example:
        UserId | GradeItemName         | PointsNumerator
        123    | Fostering             | 96
        123    | Final Calc            | 100
        456    | Fostering             | 88
    
    Becomes:
        UserId | grades_fostering | grades_final_calc
        123    | 96               | 100
        456    | 88               | NaN
    """
    if "UserId" not in df.columns:
        return pd.DataFrame(columns=["UserId"])
    
    df = df.copy()
    
    # Identify the grade item identifier column (common names)
    grade_item_col = None
    for col in ["GradeItemName", "GradeItem", "ItemName", "Item"]:
        if col in df.columns:
            grade_item_col = col
            break
    
    # Identify the grade value column - focus on numeric grades only
    grade_value_col = None
    for col in ["PointsNumerator", "Grade", "GradeValue", "Points", "Score"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            grade_value_col = col
            break
    
    if not grade_item_col or not grade_value_col:
        # Fallback to generic aggregation if structure is unexpected
        return build_user_features(df, "grades")
    
    # Select only the columns we need for pivoting
    pivot_df = df[["UserId", grade_item_col, grade_value_col]].copy()
    
    # Pivot: each grade item becomes a column
    pivot = pivot_df.pivot_table(
        index="UserId",
        columns=grade_item_col,
        values=grade_value_col,
        aggfunc="first"  # Use first value if duplicates exist
    )
    
    # Flatten column names and add prefix
    pivot.columns = [f"grades_{safe_suffix(str(item))}" for item in pivot.columns]
    pivot = pivot.reset_index()
    
    return pivot


def build_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build content progress features by pivoting topic/module interactions.
    
    Each user has multiple rows tracking their visits to different content topics/modules.
    We create features for each topic showing their engagement metrics.
    
    Example:
        UserId | ContentTopicName | TotalTime | ContentTopicVisits
        123    | Instruction      | 485518    | 2
        123    | FMP Print        | 487802    | 29
        456    | Instruction      | 400000    | 1
    
    Becomes multiple columns per topic:
        UserId | content_instruction_totaltime | content_instruction_visits | content_fmp_print_totaltime | ...
        123    | 485518                       | 2                          | 487802                      | ...
        456    | 400000                       | 1                          | NaN                         | ...
    """
    if "UserId" not in df.columns:
        return pd.DataFrame(columns=["UserId"])
    
    df = df.copy()
    
    # Identify the topic/module identifier column
    topic_col = None
    for col in ["ContentTopicName", "TopicName", "ModuleName", "ModuleID", "ContentTopic"]:
        if col in df.columns:
            topic_col = col
            break
    
    if not topic_col:
        # Fallback to generic aggregation if structure is unexpected
        return build_user_features(df, "content")
    
    # Identify value columns to pivot (exclude identifiers and UserId)
    value_cols = []
    exclude_cols = {"UserId", topic_col, "SectionId", "CourseNa"}
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            value_cols.append(col)
    
    if not value_cols:
        # Fallback if no numeric columns found
        return build_user_features(df, "content")
    
    # Pivot each value column separately, then merge
    result = df[["UserId"]].drop_duplicates().reset_index(drop=True)
    
    for val_col in value_cols:
        pivot = df.pivot_table(
            index="UserId",
            columns=topic_col,
            values=val_col,
            aggfunc="sum"  # Sum for metrics like TotalTime, ContentTopicVisits
        )
        
        # Flatten column names: content_{topic}_{metric}
        val_suffix = safe_suffix(val_col)
        pivot.columns = [f"content_{safe_suffix(str(topic))}_{val_suffix}" for topic in pivot.columns]
        pivot = pivot.reset_index()
        
        result = result.merge(pivot, on="UserId", how="left")
    
    return result


def build_eoc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build EOC survey features by pivoting survey question responses into columns.
    
    Each user completes multiple surveys (FMP, AR, Practicum), and each survey has multiple questions.
    We create columns for each (survey, question) combination to preserve all survey data.
    
    Example:
        UserId | SurveyName | QuestionNumber | Question          | Answer
        123    | FMP Survey | 1              | How satisfied...  | Very satisfied
        123    | FMP Survey | 2              | Likelihood...     | 10
        123    | AR Survey  | 1              | How useful...     | Useful
    
    Becomes:
        UserId | eoc_fmp_survey_q1_question | eoc_fmp_survey_q1_answer | eoc_ar_survey_q1_question | ...
        123    | How satisfied...           | Very satisfied           | How useful...             | ...
    """
    if "UserId" not in df.columns:
        return pd.DataFrame(columns=["UserId"])
    
    df = df.copy()
    
    # Identify the survey identifier column
    survey_col = None
    for col in ["SurveyName", "SurveyId", "CourseName"]:
        if col in df.columns:
            survey_col = col
            break
    
    if not survey_col:
        # Fallback to generic aggregation if structure is unexpected
        return build_user_features(df, "eoc")
    
    # Check for required columns
    if "QuestionNumber" not in df.columns:
        return build_user_features(df, "eoc")
    
    # Create a combined key for survey + question number
    df["survey_question_key"] = (
        df[survey_col].astype(str) + "_q" + df["QuestionNumber"].astype(str)
    )
    
    # Start with base user list
    result = df[["UserId"]].drop_duplicates().reset_index(drop=True)
    
    # Pivot Question column if it exists
    if "Question" in df.columns:
        question_pivot = df.pivot_table(
            index="UserId",
            columns="survey_question_key",
            values="Question",
            aggfunc="first"
        )
        question_pivot.columns = [f"eoc_{safe_suffix(str(key))}_question" for key in question_pivot.columns]
        question_pivot = question_pivot.reset_index()
        result = result.merge(question_pivot, on="UserId", how="left")
    
    # Pivot Answer column if it exists
    if "Answer" in df.columns:
        answer_pivot = df.pivot_table(
            index="UserId",
            columns="survey_question_key",
            values="Answer",
            aggfunc="first"
        )
        answer_pivot.columns = [f"eoc_{safe_suffix(str(key))}_answer" for key in answer_pivot.columns]
        answer_pivot = answer_pivot.reset_index()
        result = result.merge(answer_pivot, on="UserId", how="left")
    
    # Also add submission date per survey (not per question)
    if "SubmissionDate" in df.columns:
        submission_pivot = df.groupby(["UserId", survey_col])["SubmissionDate"].first().reset_index()
        submission_pivot = submission_pivot.pivot_table(
            index="UserId",
            columns=survey_col,
            values="SubmissionDate",
            aggfunc="first"
        )
        submission_pivot.columns = [f"eoc_{safe_suffix(str(survey))}_submitted" for survey in submission_pivot.columns]
        submission_pivot = submission_pivot.reset_index()
        result = result.merge(submission_pivot, on="UserId", how="left")
    
    return result


def build_user_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Generic user-level feature builder for:
        - TeacherBackgroundSurvey

    Strategy:
    - Group by UserId.
    - For each other column:
        - numeric -> mean
        - non-numeric -> unique values joined with " | "
    - Prefix all columns with dataset-specific prefix.
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
    print("Loading datasets...")
    assessments = load_csv(ASSESSMENTS_PATH)
    print(f"  Assessments: {len(assessments)} rows")
    
    grades = load_csv(GRADES_PATH)
    content = load_csv(CONTENT_PATH)
    eoc = load_csv(EOC_PATH)
    teacher = load_csv(TEACHER_PATH)

    # 1) Build per-question events from AssessmentsItemized
    print("\nProcessing assessments...")
    print(f"  Original rows (all answer choices): {len(assessments)}")
    
    selected_only = assessments[assessments["AnswerChoiceSelected"] == 1] if "AnswerChoiceSelected" in assessments.columns else assessments
    print(f"  Selected answer choices: {len(selected_only)}")
    
    events = build_assessment_events(assessments)
    print(f"  Unique question attempts: {len(events)}")
    print(f"  Unique students: {events['UserId'].nunique()}")
    print(f"  Max questions per student: {events.groupby('UserId').size().max()}")
    print(f"  Avg questions per student: {events.groupby('UserId').size().mean():.1f}")

    # 2) Pivot into wide per-student question history (q1_*, q2_*, ...)
    print("\nPivoting to wide format...")
    question_history = pivot_question_history(events)

    # 3) Build user-level aggregates from the other datasets
    print("Building user features...")
    grades_user = build_grades_features(grades)
    content_user = build_content_features(content)
    eoc_user = build_eoc_features(eoc)
    teacher_user = build_user_features(teacher, "teacher")

    # 4) Merge everything on UserId
    print("Merging all datasets...")
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
    print(f"\nSaved merged file to {OUTPUT_PATH}")
    print(f"Final shape: {merged.shape}")
    print(f"Columns: {len(merged.columns)}, Rows (students): {len(merged)}")


if __name__ == "__main__":
    main()
