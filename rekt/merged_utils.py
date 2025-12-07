import numpy as np
import pandas as pd


def count(say: str, df: pd.DataFrame,
          user_col: str = "user_id",
          problem_col: str = "problem_id",
          skill_col: str = "skill_id") -> None:
    """
    Simple dataset summary helper.

    Parameters
    ----------
    say : str
        Label printed in front of the stats.
    df : DataFrame
        Long-format interaction dataframe.
    user_col, problem_col, skill_col : str
        Column names for user, question, and concept ids.
    """
    n_rec = len(df)
    n_user = df[user_col].nunique() if user_col in df.columns else 0
    n_problem = df[problem_col].nunique() if problem_col in df.columns else 0
    n_skill = df[skill_col].nunique() if skill_col in df.columns else 0

    print(
        "%s, records: %d, students: %d, questions: %d, skills: %d"
        % (say, n_rec, n_user, n_problem, n_skill)
    )


def save_list(list_to_save, file_path: str) -> None:
    """
    Save a Python list in a very simple text form, using str(list).
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(list_to_save))


def save_dict(dict_to_save, file_path: str) -> None:
    """
    Save a Python dict in a very simple text form, using str(dict).
    This mirrors the original ReKT utilities so existing loading
    logic that uses eval still works.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(dict_to_save))


def save_graph(graph_dict, file_path: str) -> None:
    """
    Save a graph stored as an adjacency dict {src: [dst1, dst2, ...]}.

    Each line format:
        src<TAB>dst1,dst2,...
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for src, dst_list in graph_dict.items():
            if isinstance(dst_list, (list, tuple, np.ndarray, pd.Series)):
                dst_str = ",".join(str(int(d)) for d in dst_list)
            else:
                dst_str = str(dst_list)
            f.write(f"{src}\t{dst_str}\n")


def load_dict(file_path: str):
    """
    Load a dict saved with save_dict.

    This keeps the same very simple eval-based format that the
    original ASSIST09 utilities used, to stay compatible with
    downstream code that expects that behavior.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return {}
    return eval(content)


def write_lists(seq_len_list,
                questions_list,
                answers_list,
                file_path: str) -> None:
    """
    Write user sequences in the ReKT text format.

    For each user u, we write three lines:
        line 1: sequence length (int)
        line 2: comma-separated question ids
        line 3: comma-separated binary answers (0/1)

    This matches the format expected by load_data.getReader.
    """
    assert len(seq_len_list) == len(questions_list) == len(answers_list)

    with open(file_path, "w", encoding="utf-8") as f:
        for L, qs, ans in zip(seq_len_list, questions_list, answers_list):
            # defensive conversion
            L_int = int(L)
            qs_str = ",".join(str(int(q)) for q in qs)
            ans_str = ",".join(str(int(a)) for a in ans)

            f.write(f"{L_int}\n")
            f.write(f"{qs_str}\n")
            f.write(f"{ans_str}\n")


def feature_normalize(array: np.ndarray) -> np.ndarray:
    """
    Standardize a 1D or 2D numpy array to zero mean and unit variance.

    This is kept for compatibility with the original utilities and can
    be used later if you decide to add extra per-question features
    (for example, difficulty estimates) for analysis.
    """
    array = np.asarray(array, dtype=float)
    mu = np.mean(array, axis=0)
    sigma = np.std(array, axis=0)
    # avoid division by zero
    sigma[sigma == 0] = 1.0
    return (array - mu) / sigma
