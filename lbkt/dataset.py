"""
LBKT Dataset Module

Provides dataset classes for LBKT training:
- LBKTDataset: Standard dataset for BERT-style input
- LBKTSequenceDataset: DKT-compatible dataset format
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LBKTDataset(Dataset):
    """
    Dataset class for LBKT model training.

    Handles long sequences by splitting into chunks of max_seq length.
    Returns three values per sample:
    - x: Input sequence (q_id + correct * n_skill for previous interactions)
    - target_id: Target question IDs
    - label: Target correctness labels
    """

    def __init__(self, group, n_skill, min_samples=1, max_seq=100):
        """
        Args:
            group: Dict-like {user_id: (question_ids, correctness)}
            n_skill: Number of unique skills/questions
            min_samples: Minimum sequence length to include
            max_seq: Maximum sequence length (longer sequences are chunked)
        """
        super(LBKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        self.user_ids = []

        for user_id in group.index if hasattr(group, 'index') else group.keys():
            q, qa = group[user_id]  # q: question IDs, qa: correctness (0/1)
            
            if len(q) < min_samples:
                continue

            # Handle long sequences by chunking
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                
                # First chunk (remainder)
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial])
                
                # Full chunks
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = start + self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], qa[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        # Initialize with zeros (padding)
        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        
        # Fill from the end (right-aligned padding)
        if seq_len == self.max_seq:
            q[:] = q_
            qa[:] = qa_
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_

        # Target: next question prediction (shifted by 1)
        target_id = q[1:]
        label = qa[1:]

        # Input: current question + previous answer
        # Encode as q_id + (correct * n_skill)
        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label


class LBKTSequenceDataset(Dataset):
    """
    DKT-compatible dataset format for LBKT.

    Returns (q_seq, r_seq, mask, user_id, completion_rate) for each sample where:
    - q_seq: Question ID sequence
    - r_seq: Response (correctness) sequence
    - mask: Valid position mask
    - user_id: User identifier
    - completion_rate: User's completion rate for fairness analysis
    """

    def __init__(
        self,
        pkl_path,
        max_len=310,
        completion_rates=None,
    ):
        """
        Args:
            pkl_path: Path to pickle file with list of ((q_arr, r_arr), user_id) tuples
            max_len: Maximum sequence length (pad/truncate)
            completion_rates: Optional dict {user_id: rate} for fairness analysis
        """
        import pickle
        
        with open(pkl_path, 'rb') as f:
            self.sequences = pickle.load(f)
        
        self.max_len = max_len
        self.completion_rates = completion_rates or {}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        (q_seq, r_seq), user_id = self.sequences[idx]
        seq_len = len(q_seq)

        # Convert to numpy if needed
        if not isinstance(q_seq, np.ndarray):
            q_seq = np.array(q_seq, dtype=np.int64)
        if not isinstance(r_seq, np.ndarray):
            r_seq = np.array(r_seq, dtype=np.int64)

        # Truncate if needed
        if seq_len > self.max_len:
            q_seq = q_seq[-self.max_len:]
            r_seq = r_seq[-self.max_len:]
            seq_len = self.max_len

        # Pad sequences
        q_padded = np.zeros(self.max_len, dtype=np.int64)
        r_padded = np.zeros(self.max_len, dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=np.float32)

        q_padded[:seq_len] = q_seq
        r_padded[:seq_len] = r_seq
        mask[:seq_len] = 1.0
        
        # Get completion rate for this user
        cr = self.completion_rates.get(user_id, 0.5)

        return (
            torch.from_numpy(q_padded),
            torch.from_numpy(r_padded),
            torch.from_numpy(mask),
            str(user_id),
            cr,
        )


def create_lbkt_dataloader(
    pkl_path,
    max_len,
    batch_size,
    shuffle=True,
    num_workers=0,
    completion_rates=None,
):
    """
    Create a DataLoader for LBKT training.

    Args:
        pkl_path: Path to pickle file
        max_len: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        completion_rates: Optional completion rates for fairness analysis

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    dataset = LBKTSequenceDataset(
        pkl_path=pkl_path,
        max_len=max_len,
        completion_rates=completion_rates,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
