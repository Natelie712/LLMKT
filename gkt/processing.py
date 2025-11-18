
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import build_dense_graph


# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com
#
# This file has been lightly refactored to:
#   - work with your merged long-format CSV (user_id, skill_id, correct)
#   - keep the original interface used by train.py (load_dataset)
#   - keep Dense / Transition / DKT graph construction behavior
# The core model code in models.py and loss code in metrics.py are unchanged.


TOTAL_Q_SLOTS = 305  # maximum question attempt slots in the assessment bank


class KTDataset(Dataset):
    """
    Simple container for per-student sequences.

    Each item is a triple of Python lists:
        features[t]  = skill_with_answer index at time t
        questions[t] = skill index at time t
        answers[t]   = 0/1 correctness at time t
    """

    def __init__(self, features, questions, answers, user_ids, completion_rates):
        assert len(features) == len(questions) == len(answers)
        assert len(user_ids) == len(features)
        assert len(completion_rates) == len(features)
        self.features = features
        self.questions = questions
        self.answers = answers
        self.user_ids = user_ids
        self.completion_rates = completion_rates

    def __getitem__(self, index):
        return (
            self.features[index],
            self.questions[index],
            self.answers[index],
            self.user_ids[index],
            self.completion_rates[index],
        )

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    """
    Collate function that pads variable-length sequences with -1.

    This is consistent with:
      - models.GKT, which uses qt == -1 as padding mask,
      - metrics.KTLoss, which ignores real_answers == -1.
    """
    features, questions, answers, user_ids, completion_rates = zip(*batch)

    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]

    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)

    # user_ids: keep as a list (no padding)
    # completion_rates: convert to numpy array for easy indexing
    comp_rates = np.array(completion_rates, dtype=float)
    return feature_pad, question_pad, answer_pad, list(user_ids), comp_rates


def load_dataset(
    file_path,
    batch_size,
    graph_type,
    dkt_graph_path=None,
    train_ratio=0.7,
    val_ratio=0.1,
    shuffle=True,
    model_type="GKT",
    use_binary=True,
    res_len=2,
    use_cuda=True,
):
    r"""
    Parameters
    ----------
    file_path : str
        Path to long-format knowledge tracing CSV. Must contain:
          - user_id
          - skill_id
          - correct (0/1)
    batch_size : int
        Batch size in students.
    graph_type : str
        One of ['Dense', 'Transition', 'DKT', 'PAM', 'MHA', 'VAE'].
        For PAM / MHA / VAE, we do not construct a static graph here.
    dkt_graph_path : str or None
        Path to a precomputed DKT graph text file (for graph_type == 'DKT').
    train_ratio : float
        Fraction of students for training.
    val_ratio : float
        Fraction of students for validation.
    shuffle : bool
        Whether to shuffle students before splitting.
    model_type : str
        'GKT' or 'DKT'. Only 'GKT' uses graphs built here.
    use_binary : bool
        If True, skill_with_answer = skill * 2 + correct.
        If False, skill_with_answer = skill * res_len + correct - 1.
    res_len : int
        Number of response categories (usually 2).
    use_cuda : bool
        Kept for interface compatibility. Graph is moved to CUDA here if True.

    Returns
    -------
    concept_num : int
        Number of unique concepts/questions.
    graph : torch.Tensor or None
        Static concept graph if graph_type in ['Dense', 'Transition', 'DKT']
        and model_type == 'GKT'; otherwise None.
    train_loader, valid_loader, test_loader : DataLoader
        Data loaders over KTDataset with pad_collate.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Basic column checks
    for col in ["skill_id", "correct", "user_id"]:
        if col not in df.columns:
            raise KeyError(f"The column '{col}' was not found in {file_path}")

    # Ensure 'correct' is 0/1 integers
    df["correct"] = df["correct"].astype(float)
    # Clip / round to {0,1} just in case
    df["correct"] = df["correct"].round().clip(0, 1).astype(int)

    # Step 1.1: Remove questions without skill
    df = df.dropna(subset=["skill_id"])

    # Step 1.2: Remove users with a single answer
    df = df.groupby("user_id").filter(lambda g: len(g) > 1).copy()

    # Step 2: Enumerate skill_id to contiguous integer indices
    df["skill"], skill_categories = pd.factorize(df["skill_id"], sort=True)
    question_dim = int(df["skill"].max() + 1)
    concept_num = question_dim

    # Step 3: Cross skill with answer to form synthetic feature
    if use_binary:
        # skill * 2 + correct ∈ [0, 2 * concept_num - 1]
        df["skill_with_answer"] = df["skill"] * 2 + df["correct"]
    else:
        # skill * res_len + (correct - 1) ∈ [0, res_len * concept_num - 1]
        df["skill_with_answer"] = df["skill"] * res_len + df["correct"] - 1

    # Step 4: Aggregate sequences per user_id
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    user_ids_list = []
    completion_rates = []

    def collect_user(series: pd.DataFrame):
        feature_list.append(series["skill_with_answer"].tolist())
        question_list.append(series["skill"].tolist())
        # Convert correctness back to 0/1 for training
        answer_list.append(series["correct"].astype(int).tolist())
        L = int(series["correct"].shape[0])
        seq_len_list.append(L)
        uid = series["user_id"].iloc[0]
        user_ids_list.append(uid)
        completion_rates.append(float(L) / float(TOTAL_Q_SLOTS) if TOTAL_Q_SLOTS > 0 else 0.0)

    df.groupby("user_id").apply(collect_user)

    max_seq_len = int(np.max(seq_len_list)) if seq_len_list else 0
    student_num = len(seq_len_list)
    feature_dim = int(df["skill_with_answer"].max() + 1) if len(df) > 0 else 0

    print(f"max_seq_len: {max_seq_len}")
    print(f"student num: {student_num}")
    print(f"feature_dim: {feature_dim}")
    print(f"question_dim: {question_dim}")

    # Build dataset of all students
    kt_dataset = KTDataset(feature_list, question_list, answer_list, user_ids_list, completion_rates)

    # Train/val/test split by student index
    train_size = int(student_num * train_ratio)
    val_size = int(student_num * val_ratio)
    test_size = student_num - train_size - val_size
    print(f"train_size: {train_size}  val_size: {val_size}  test_size: {test_size}")

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        kt_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )

    # Build concept graph (if needed)
    graph = None
    if model_type == "GKT":
        if graph_type == "Dense":
            graph = build_dense_graph(concept_num)
        elif graph_type == "Transition":
            # Transition graph is built from training sequences only
            graph = build_transition_graph(
                question_list, seq_len_list, train_dataset.indices, student_num, concept_num
            )
        elif graph_type == "DKT":
            if dkt_graph_path is None:
                raise ValueError("dkt_graph_path must be provided when graph_type == 'DKT'")
            graph = build_dkt_graph(dkt_graph_path, concept_num)

    if graph is not None and use_cuda and torch.cuda.is_available():
        graph = graph.cuda()

    return concept_num, graph, train_loader, valid_loader, test_loader


def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    """
    Build a row-normalized transition graph:
        graph[u, v] ∝ number of transitions u -> v in the training data.
    """
    graph = np.zeros((concept_num, concept_num))
    # Map dataset subset indices to [0, student_num)
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            nxt = questions[j + 1]
            graph[pre, nxt] += 1

    np.fill_diagonal(graph, 0)

    # Row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1.0 / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)

    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    """
    Load a precomputed DKT-style graph from a text file.
    """
    graph = np.loadtxt(file_path)
    assert (
        graph.shape[0] == concept_num and graph.shape[1] == concept_num
    ), "DKT graph shape does not match concept_num"
    graph = torch.from_numpy(graph).float()
    return graph
