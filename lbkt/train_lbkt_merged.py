"""
LBKT Training Script for Merged Dataset

Train and evaluate LBKT on sequences produced by lbkt_preprocess_merged.py.

Usage (from lbkt/ directory, after running lbkt_preprocess_merged.py):

    python train_lbkt_merged.py \
        --data_dir lbkt_processed \
        --max_len 100 \
        --embed_dim 128 \
        --num_heads 4 \
        --num_layers 2 \
        --lstm_hidden 128 \
        --lstm_layers 1 \
        --dropout 0.1 \
        --batch_size 64 \
        --epochs 30 \
        --lr 1e-3 \
        --device cuda

This script:
  * Loads LBKT-ready pickles (train/valid/test) and metadata.
  * Wraps them in Dataset/DataLoader via dataset.py.
  * Trains the LBKT model (BERT + Rasch embeddings + LSTM).
  * Reports AUC, ACC, and F1 for valid/test.
  * Computes fairness metrics on completion-rate bins during eval.
"""

import argparse
import json
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from model import LBKT, LBKTSimple
from dataset import LBKTSequenceDataset


# --------------------
# Utilities
# --------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    """Find optimal threshold by sweeping from 0.05 to 0.95."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    
    best_score = -1.0
    best_t = 0.5
    
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (y_probs >= t).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'acc':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'balanced_acc':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_t = t
    
    return float(best_t), float(best_score)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LBKT on merged dataset")

    # Data
    parser.add_argument("--data_dir", type=str, default="lbkt_processed",
                        help="Directory with lbkt_train/valid/test.pkl and metadata.")
    parser.add_argument("--max_len", type=int, default=310,
                        help="Max sequence length (pad/truncate).")
    
    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers.")
    parser.add_argument("--lstm_hidden", type=int, default=64,
                        help="LSTM hidden size.")
    parser.add_argument("--lstm_layers", type=int, default=1,
                        help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate.")
    parser.add_argument("--model_type", type=str, default="lbkt",
                        choices=["lbkt", "lbkt_simple"],
                        help="Model type: 'lbkt' (full) or 'lbkt_simple' (lightweight).")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW.")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Number of warmup epochs.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers.")
    parser.add_argument("--save_best", type=str, default="lbkt_best.pt",
                        help="Path to save best model (by valid AUC).")

    return parser.parse_args()


# --------------------
# Metrics
# --------------------

def compute_overall_metrics(all_logits: List[float], all_labels: List[int]) -> Dict[str, float]:
    """Compute overall AUC, ACC, and F1 given flat predictions and labels."""
    y_true = np.array(all_labels, dtype=np.int32)
    y_pred = np.array(all_logits, dtype=np.float32)

    metrics = {}

    # Accuracy (threshold 0.5)
    y_hat = (y_pred >= 0.5).astype(np.int32)
    acc = (y_hat == y_true).mean() if y_true.size > 0 else float("nan")
    metrics["acc"] = float(acc)

    # AUC
    if y_true.size > 0 and np.unique(y_true).size == 2:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")
    metrics["auc"] = float(auc)

    # F1 Score
    if y_true.size > 0 and np.unique(y_true).size == 2:
        try:
            f1 = f1_score(y_true, y_hat)
        except Exception:
            f1 = float("nan")
    else:
        f1 = float("nan")
    metrics["f1"] = float(f1)

    return metrics


def bin_index_from_completion_rate(cr: float, num_bins: int = 10) -> int:
    """Map completion rate in [0,1] to bin index [0, num_bins-1]."""
    if cr < 0.0:
        cr = 0.0
    if cr > 1.0:
        cr = 1.0
    if cr >= 1.0:
        return num_bins - 1
    return int(cr * num_bins)


def compute_fairness_metrics(
    bin_stats: Dict[int, Dict[str, List[float]]],
) -> Dict[str, float]:
    """Compute fairness metrics from binned predictions and labels."""
    per_bin = {}
    bin_indices = sorted(bin_stats.keys())

    for b in bin_indices:
        preds = np.array(bin_stats[b]["preds"], dtype=np.float32)
        labels = np.array(bin_stats[b]["labels"], dtype=np.int32)
        if labels.size == 0:
            continue
        y_hat = (preds >= 0.5).astype(np.int32)

        # Confusion components
        tp = np.sum((y_hat == 1) & (labels == 1))
        tn = np.sum((y_hat == 0) & (labels == 0))
        fp = np.sum((y_hat == 1) & (labels == 0))
        fn = np.sum((y_hat == 0) & (labels == 1))
        total = labels.size

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        acc = (tp + tn) / total if total > 0 else np.nan
        
        # Precision and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else np.nan
        
        # AUC for this bin
        if np.unique(labels).size == 2:
            try:
                auc = roc_auc_score(labels, preds)
            except:
                auc = np.nan
        else:
            auc = np.nan

        num_students = len(bin_stats[b].get("users", set()))
        
        per_bin[b] = {
            "tpr": float(tpr),
            "fpr": float(fpr),
            "acc": float(acc),
            "f1": float(f1),
            "auc": float(auc),
            "count": int(total),
            "num_students": num_students,
        }

    # Print bin-level stats
    print("  Fairness per completion-rate bin:")
    for b in sorted(per_bin.keys()):
        info = per_bin[b]
        lo = 10 * b
        hi = 10 * (b + 1)
        print(
            f"    Bin {b} ({lo:2d}-{hi:3d}%): "
            f"students={info['num_students']}, predictions={info['count']}, "
            f"TPR={info['tpr']:.3f}, FPR={info['fpr']:.3f}, ACC={info['acc']:.3f}, "
            f"F1={info['f1']:.3f}, AUC={info['auc']:.3f}"
        )

    # Equalized odds distance between lowest and highest non-empty bins
    if per_bin:
        low_b = min(per_bin.keys())
        high_b = max(per_bin.keys())
        low = per_bin[low_b]
        high = per_bin[high_b]

        if not (np.isnan(low["tpr"]) or np.isnan(low["fpr"]) or
                np.isnan(high["tpr"]) or np.isnan(high["fpr"])):
            eo = float(
                np.sqrt(
                    (low["tpr"] - high["tpr"]) ** 2
                    + (low["fpr"] - high["fpr"]) ** 2
                )
            )
        else:
            eo = float("nan")
    else:
        eo = float("nan")

    # Variance metrics
    if per_bin:
        low_b = min(per_bin.keys())
        high_b = max(per_bin.keys())
        low = per_bin[low_b]
        high = per_bin[high_b]
        
        acc_var = abs(low["acc"] - high["acc"]) if not (np.isnan(low["acc"]) or np.isnan(high["acc"])) else float("nan")
        f1_var = abs(low["f1"] - high["f1"]) if not (np.isnan(low["f1"]) or np.isnan(high["f1"])) else float("nan")
        auc_var = abs(low["auc"] - high["auc"]) if not (np.isnan(low["auc"]) or np.isnan(high["auc"])) else float("nan")
    else:
        acc_var = float("nan")
        f1_var = float("nan")
        auc_var = float("nan")

    print(f"  Equalized odds distance (lowest vs highest non-empty bin): {eo:.4f}")
    print(f"  Difference (lowest vs highest bin) - ACC: {acc_var:.6f}, F1: {f1_var:.6f}, AUC: {auc_var:.6f}")

    return {
        "eo_low_high": eo,
        "acc_var": acc_var,
        "f1_var": f1_var,
        "auc_var": auc_var,
    }


# --------------------
# Training and eval
# --------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train: bool = True,
    optimizer=None,
    scheduler=None,
):
    """Run one train/validation/test epoch.

    Returns:
      loss_avg, metrics_dict, fairness_dict
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    all_logits: List[float] = []
    all_labels: List[int] = []

    # For fairness: bin -> {preds: [...], labels: [...], users: set()}
    bin_stats: Dict[int, Dict] = {}

    criterion = nn.BCEWithLogitsLoss(reduction='none')

    for batch in loader:
        # Handle different batch formats based on model type
        # Standard dataset returns: q, r, mask, users, completion_rates
        # (lbkt_text now uses pre-computed embeddings so same dataset format)
        q, r, mask, users, completion_rates = batch
        q = q.to(device)
        r = r.to(device)
        mask = mask.to(device)

        if train:
            optimizer.zero_grad()

        # Forward - handle different model types
        if model.model_name == "lbkt":
            # LBKT uses shifted inputs as in the original BERT-LSTM notebook:
            # - x: encoded history q[:-1] + (r[:-1]==1) * num_questions
            # - segment_info: target question IDs q[1:]
            # - output predicts r[1:] directly (no additional shifting needed)
            
            # Shift inputs: history from positions 0 to T-2
            q_history = q[:, :-1]  # [B, T-1]
            r_history = r[:, :-1]  # [B, T-1]
            
            # Encode history: q_id + correct * num_questions
            r_history_long = r_history.long()
            x_encoded = q_history + model.num_questions * r_history_long  # [B, T-1]
            
            # Target questions (segment_info): positions 1 to T-1
            segment_info = q[:, 1:]  # [B, T-1]
            
            # Forward pass - output directly predicts next-step correctness
            logits = model(x_encoded, segment_info)  # shape [B, T-1]
            
            # Labels and masks for next positions
            logits_next = logits  # [B, T-1] - already aligned with next-step
            r_next = r[:, 1:]  # [B, T-1]
            m_next = mask[:, 1:]  # [B, T-1]
        else:
            # LBKTSimple takes raw q, r and encodes internally
            y_full = model(q, r)  # shape [B, T, num_questions] with sigmoid
            # Gather predictions for next questions
            q_next = q[:, 1:]  # [B, T-1]
            y_next_full = y_full[:, :-1, :]  # [B, T-1, num_questions]
            # Gather predictions at each position for the actual next question
            idx = q_next.unsqueeze(-1)  # [B, T-1, 1]
            logits_next = torch.gather(y_next_full, dim=2, index=idx).squeeze(-1)  # [B, T-1]
            # Convert back to logits for BCE loss (inverse sigmoid)
            logits_next = torch.log(logits_next / (1 - logits_next + 1e-8) + 1e-8)
            r_next = r[:, 1:]  # [B, T-1]
            m_next = mask[:, 1:]  # [B, T-1]

        # Compute loss only on valid positions
        loss_unreduced = criterion(logits_next, r_next)
        loss_masked = loss_unreduced * m_next
        
        if m_next.sum() > 0:
            loss = loss_masked.sum() / m_next.sum()
        else:
            continue

        count = m_next.sum().item()
        total_loss += loss.item() * count
        total_count += count

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Collect metrics (apply sigmoid to logits for probabilities)
        probs = torch.sigmoid(logits_next)
        
        B = q.size(0)
        for i in range(B):
            m_i = m_next[i] > 0.0
            if m_i.sum().item() == 0:
                continue

            p_i = probs[i][m_i]
            r_i = r_next[i][m_i]

            all_logits.extend(p_i.detach().cpu().numpy().tolist())
            all_labels.extend(r_i.detach().cpu().numpy().astype(int).tolist())

            # Fairness binning
            cr = float(completion_rates[i])
            b = bin_index_from_completion_rate(cr, num_bins=10)
            if b not in bin_stats:
                bin_stats[b] = {"preds": [], "labels": [], "users": set()}

            bin_stats[b]["users"].add(users[i])
            bin_stats[b]["preds"].extend(p_i.detach().cpu().numpy().tolist())
            bin_stats[b]["labels"].extend(r_i.detach().cpu().numpy().astype(int).tolist())

    if total_count > 0:
        avg_loss = total_loss / float(total_count)
    else:
        avg_loss = float("nan")

    metrics = compute_overall_metrics(all_logits, all_labels)

    # Fairness only meaningful for eval
    fairness = {}
    if not train:
        fairness = compute_fairness_metrics(bin_stats)

    return avg_loss, metrics, fairness


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Create a schedule with linear warmup and linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# --------------------
# Main
# --------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = args.device.lower().startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    meta_path = os.path.join(args.data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_questions = int(meta["num_questions"])
    total_q_slots = float(meta["total_q_slots"])
    print(f"Metadata: num_questions={num_questions}, total_q_slots={total_q_slots}")

    # Load completion rates for fairness analysis
    completion_rates_path = os.path.join(args.data_dir, "completion_rates.pkl")
    if os.path.exists(completion_rates_path):
        with open(completion_rates_path, "rb") as f:
            completion_rates = pickle.load(f)
    else:
        completion_rates = {}

    # Datasets and loaders
    train_ds = LBKTSequenceDataset(
        os.path.join(args.data_dir, "lbkt_train.pkl"),
        max_len=args.max_len,
        completion_rates=completion_rates,
    )
    valid_ds = LBKTSequenceDataset(
        os.path.join(args.data_dir, "lbkt_valid.pkl"),
        max_len=args.max_len,
        completion_rates=completion_rates,
    )
    test_ds = LBKTSequenceDataset(
        os.path.join(args.data_dir, "lbkt_test.pkl"),
        max_len=args.max_len,
        completion_rates=completion_rates,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train: {len(train_ds)} users, Valid: {len(valid_ds)} users, Test: {len(test_ds)} users")

    # Model
    if args.model_type == "lbkt":
        model = LBKT(
            num_questions=num_questions,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            max_seq_len=args.max_len,
            dropout=args.dropout,
        ).to(device)
    else:
        model = LBKTSimple(
            num_questions=num_questions,
            embed_dim=args.embed_dim,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            max_seq_len=args.max_len,
            dropout=args.dropout,
        ).to(device)

    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_valid_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_metrics, _ = run_epoch(
            model, train_loader, device, train=True, optimizer=optimizer, scheduler=scheduler
        )
        print(
            f"  [Train] loss={train_loss:.4f}, "
            f"AUC={train_metrics['auc']:.4f}, ACC={train_metrics['acc']:.4f}, F1={train_metrics['f1']:.4f}"
        )

        # Validation
        with torch.no_grad():
            valid_loss, valid_metrics, valid_fairness = run_epoch(
                model, valid_loader, device, train=False
            )
        print(
            f"  [Valid] loss={valid_loss:.4f}, "
            f"AUC={valid_metrics['auc']:.4f}, ACC={valid_metrics['acc']:.4f}, F1={valid_metrics['f1']:.4f}"
        )
        print(
            f"          EO(low-high)={valid_fairness.get('eo_low_high', float('nan')):.4f}, "
            f"F1 var={valid_fairness.get('f1_var', float('nan')):.6f}, "
            f"AUC var={valid_fairness.get('auc_var', float('nan')):.6f}"
        )

        # Track best by validation AUC
        valid_auc = valid_metrics["auc"]
        if valid_auc is not None and valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "valid_auc": valid_auc,
                "args": vars(args),
                "meta": meta,
            }
            torch.save(best_state, args.save_best)
            print(f"  [Info] New best model saved to {args.save_best} (valid AUC={valid_auc:.4f})")

    # Load best model if available
    if best_state is not None:
        print(f"\nLoading best model from {args.save_best} (valid AUC={best_state['valid_auc']:.4f})")
        model.load_state_dict(best_state["model_state"])
    else:
        print("No valid best model state found; using final epoch weights.")

    # Final test evaluation
    with torch.no_grad():
        test_loss, test_metrics, test_fairness = run_epoch(
            model, test_loader, device, train=False
        )

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(
        f"[Test] loss={test_loss:.4f}, "
        f"AUC={test_metrics['auc']:.4f}, ACC={test_metrics['acc']:.4f}, F1={test_metrics['f1']:.4f}"
    )
    print(
        f"       EO(low-high)={test_fairness.get('eo_low_high', float('nan')):.4f}, "
        f"F1 var={test_fairness.get('f1_var', float('nan')):.6f}, "
        f"AUC var={test_fairness.get('auc_var', float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
