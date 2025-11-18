
# dkt_dataset.py
# Dataset + DataLoader utilities for DKT after preprocessing.

import json
import pickle
import torch
from torch.utils.data import Dataset

class DKTSequenceDataset(Dataset):
    def __init__(self, pkl_path, max_len=300, device="cpu"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.users = data["users"]
        self.q_seqs = data["q_seqs"]
        self.r_seqs = data["r_seqs"]
        self.seq_lengths = data["seq_lengths"]
        # Completion rates: STABLE per student, precomputed in preprocessing
        # Formula: completion_rate = student_questions / TOTAL_Q_SLOTS
        # Used for fairness binning (same student -> same bin always)
        self.completion_rates = data["completion_rates"]
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        q = self.q_seqs[idx]
        r = self.r_seqs[idx]
        L = len(q)

        T = min(L, self.max_len)

        q_pad = torch.zeros(self.max_len, dtype=torch.long)
        r_pad = torch.zeros(self.max_len, dtype=torch.float32)
        mask = torch.zeros(self.max_len, dtype=torch.float32)

        q_pad[:T] = torch.tensor(q[:T], dtype=torch.long)
        r_pad[:T] = torch.tensor(r[:T], dtype=torch.float32)
        mask[:T] = 1.0

        return (
            q_pad.to(self.device),
            r_pad.to(self.device),
            mask.to(self.device),
            self.users[idx],
            self.completion_rates[idx],  # Stable precomputed rate for fairness
        )

def load_dkt_splits(base_dir, max_len=300, device="cpu"):
    from torch.utils.data import DataLoader

    train = DKTSequenceDataset(f"{base_dir}/dkt_train.pkl", max_len, device)
    valid = DKTSequenceDataset(f"{base_dir}/dkt_valid.pkl", max_len, device)
    test = DKTSequenceDataset(f"{base_dir}/dkt_test.pkl", max_len, device)

    return train, valid, test
