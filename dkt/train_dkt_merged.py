
# train_dkt_merged.py
# Train and evaluate DKT on sequences produced by dkt_preprocess_merged.py.
#
# Usage (from repo root, after running dkt_preprocess_merged.py):
#
#   python train_dkt_merged.py \
#       --data_dir dkt_processed \
#       --max_len 300 \
#       --hidden 200 \
#       --dropout 0.1 \
#       --batch_size 64 \
#       --epochs 30 \
#       --lr 1e-3 \
#       --device cuda
#
# This script:
#   * Loads DKT-ready pickles (train/valid/test) and metadata.
#   * Wraps them in Dataset/DataLoader via dkt_dataset.py.
#   * Trains your DKT model (from dkt.py) with BCE loss.
#   * Reports AUC, ACC, and F1 for valid/test.
#   * Computes fairness metrics on completion-rate bins during eval only:
#       - Uses STABLE precomputed completion rates (not batch-dependent)
#       - Bins students by engagement level (0-10%, 10-20%, ..., 90-100%)
#       - Per-bin TPR/FPR/ACC
#       - Equalized odds distance (lowest vs highest non-empty bin)
#       - Accuracy variance across bins
#       - Same student ALWAYS maps to same bin regardless of batch

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader

from dkt import DKT
from dkt_dataset import DKTSequenceDataset


# --------------------
# Utilities
# --------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DKT on merged dataset")

    parser.add_argument("--data_dir", type=str, default="dkt_processed",
                        help="Directory with dkt_train/valid/test.pkl and metadata.")
    parser.add_argument("--max_len", type=int, default=300,
                        help="Max sequence length (pad/truncate).")
    parser.add_argument("--hidden", type=int, default=200,
                        help="Embedding/hidden size for DKT.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers.")
    parser.add_argument("--save_best", type=str, default="dkt_best.pt",
                        help="Path to save best model (by valid AUC).")

    return parser.parse_args()


# --------------------
# Metrics
# --------------------

def compute_overall_metrics(all_logits: List[float], all_labels: List[int]) -> Dict[str, float]:
    """Compute overall AUC and ACC given flat predictions and labels.

    all_logits: list or 1D array of predicted probabilities in [0,1].
    all_labels: list or 1D array of 0/1 labels.
    """
    y_true = np.array(all_labels, dtype=np.int32)
    y_pred = np.array(all_logits, dtype=np.float32)

    metrics = {}

    # Accuracy (threshold 0.5)
    y_hat = (y_pred >= 0.5).astype(np.int32)
    acc = (y_hat == y_true).mean() if y_true.size > 0 else float("nan")
    metrics["acc"] = float(acc)

    # AUC (guard against degenerate all-0 or all-1 labels)
    if y_true.size > 0 and np.unique(y_true).size == 2:
        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")
    metrics["auc"] = float(auc)

    # F1 Score (guard against degenerate all-0 or all-1 labels)
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
    """Map completion rate in [0,1] to bin index [0, num_bins-1].
    
    IMPORTANT: cr is a STABLE student property from preprocessing:
        cr = (student's total questions) / TOTAL_Q_SLOTS
    This ensures the same student always maps to the same bin.
    
    Args:
        cr: Precomputed completion rate from preprocessing (stable per student)
        num_bins: Number of engagement bins (default 10 for deciles)
    
    Returns:
        Bin index in [0, num_bins-1] representing engagement group
    """
    if cr < 0.0:
        cr = 0.0
    if cr > 1.0:
        cr = 1.0
    # Special case exact 1.0 to last bin
    if cr >= 1.0:
        return num_bins - 1
    return int(cr * num_bins)


def compute_fairness_metrics(
    bin_stats: Dict[int, Dict[str, List[float]]],
) -> Dict[str, float]:
    """Compute fairness metrics from binned predictions and labels.

    bin_stats: dict[bin_idx] -> {"preds": [...], "labels": [...]}
    Returns:
      {
        "eo_low_high": float or NaN,
        "acc_var": float or NaN,
      }
    and prints a short summary.
    """
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

        # Count unique students in this bin
        num_students = len(bin_stats[b].get("users", set()))
        
        per_bin[b] = {
            "tpr": float(tpr),
            "fpr": float(fpr),
            "acc": float(acc),
            "count": int(total),  # Total predictions
            "num_students": num_students,  # Unique students
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
            f"TPR={info['tpr']:.3f}, FPR={info['fpr']:.3f}, ACC={info['acc']:.3f}"
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

    # Accuracy variance across bins (over those with non-NaN ACC)
    accs = [info["acc"] for info in per_bin.values() if not np.isnan(info["acc"])]
    if len(accs) > 0:
        acc_var = float(np.var(np.array(accs, dtype=np.float32)))
    else:
        acc_var = float("nan")

    print(f"  Equalized odds distance (lowest vs highest non-empty bin): {eo:.4f}")
    print(f"  Accuracy variance across bins: {acc_var:.6f}")

    return {
        "eo_low_high": eo,
        "acc_var": acc_var,
    }


# --------------------
# Training and eval
# --------------------

def run_epoch(
    model: DKT,
    loader: DataLoader,
    device: torch.device,
    train: bool = True,
    optimizer=None,
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

    for batch in loader:
        q, r, mask, users, completion_rates = batch
        # q, r, mask are already on device in our Dataset __getitem__

        if train:
            optimizer.zero_grad()

        # Forward
        # Model outputs y: [B, T, num_c] with probabilities in [0,1]
        y_full = model(q, r)  # shape [B, T, num_c]

        # Next-step prediction setup:
        # We use predictions at time t (y_full[:, t, :]) to predict r at t+1.
        # So we align as:
        #   q_next = q[:, 1:]
        #   r_next = r[:, 1:]
        #   m_next = mask[:, 1:]
        #   y_next_full = y_full[:, :-1, :]
        q_next = q[:, 1:]
        r_next = r[:, 1:]
        m_next = mask[:, 1:]
        y_next_full = y_full[:, :-1, :]

        B, Tm1 = q_next.shape
        num_c = y_next_full.size(-1)

        # Gather predictions for the actual next question
        idx = q_next.unsqueeze(-1)  # [B, T-1, 1]
        y_next = torch.gather(y_next_full, dim=2, index=idx).squeeze(-1)  # [B, T-1]

        # Flatten, but only keep positions where m_next == 1
        mask_flat = m_next.reshape(-1) > 0.0
        if mask_flat.sum() == 0:
            # No valid steps in this batch (unlikely), skip
            continue

        y_flat = y_next.reshape(-1)[mask_flat]
        r_flat = r_next.reshape(-1)[mask_flat]

        # BCE loss (model already outputs sigmoid probs)
        loss = F.binary_cross_entropy(y_flat, r_flat)

        count = mask_flat.sum().item()
        total_loss += loss.item() * count
        total_count += count

        if train:
            loss.backward()
            optimizer.step()

        # Collect for overall metrics (detach to CPU)
        all_logits.extend(y_flat.detach().cpu().numpy().tolist())
        all_labels.extend(r_flat.detach().cpu().numpy().astype(int).tolist())

        # Collect bin-level stats for fairness using STABLE completion rates
        # CRITICAL: completion_rates[i] is PRECOMPUTED in preprocessing
        # Formula: cr = (student_questions / TOTAL_Q_SLOTS)
        # This ensures same student -> same bin across ALL batches
        for i in range(B):
            cr = float(completion_rates[i])  # Stable precomputed rate
            b = bin_index_from_completion_rate(cr, num_bins=10)
            if b not in bin_stats:
                bin_stats[b] = {"preds": [], "labels": [], "users": set()}

            # Track unique user in this bin
            bin_stats[b]["users"].add(users[i])

            # Valid positions for this user in this batch
            m_i = m_next[i] > 0.0
            if m_i.sum().item() == 0:
                continue

            y_i = y_next[i][m_i]
            r_i = r_next[i][m_i]

            bin_stats[b]["preds"].extend(y_i.detach().cpu().numpy().tolist())
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
    meta_path = os.path.join(args.data_dir, "dkt_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_questions = int(meta["num_questions"])
    total_q_slots = float(meta["total_q_slots"])
    print(f"Metadata: num_questions={num_questions}, total_q_slots={total_q_slots}")

    # Datasets and loaders
    train_ds = DKTSequenceDataset(
        os.path.join(args.data_dir, "dkt_train.pkl"),
        max_len=args.max_len,
        device=device,
    )
    valid_ds = DKTSequenceDataset(
        os.path.join(args.data_dir, "dkt_valid.pkl"),
        max_len=args.max_len,
        device=device,
    )
    test_ds = DKTSequenceDataset(
        os.path.join(args.data_dir, "dkt_test.pkl"),
        max_len=args.max_len,
        device=device,
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

    # Model
    model = DKT(
        num_c=num_questions,
        emb_size=args.hidden,
        dropout=args.dropout,
        emb_type="qid",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_metrics, _ = run_epoch(
            model, train_loader, device, train=True, optimizer=optimizer
        )
        print(
            f"  [Train] loss={train_loss:.4f}, "
            f"AUC={train_metrics['auc']:.4f}, ACC={train_metrics['acc']:.4f}, F1={train_metrics['f1']:.4f}"
        )

        # Validation
        with torch.no_grad():
            valid_loss, valid_metrics, valid_fairness = run_epoch(
                model, valid_loader, device, train=False, optimizer=None
            )
        print(
            f"  [Valid] loss={valid_loss:.4f}, "
            f"AUC={valid_metrics['auc']:.4f}, ACC={valid_metrics['acc']:.4f}, F1={valid_metrics['f1']:.4f}"
        )
        print(
            f"          EO(low-high)={valid_fairness.get('eo_low_high', float('nan')):.4f}, "
            f"ACC var={valid_fairness.get('acc_var', float('nan')):.6f}"
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
        print(f"Loading best model from {args.save_best} (valid AUC={best_state['valid_auc']:.4f})")
        model.load_state_dict(best_state["model_state"])
    else:
        print("No valid best model state found; using final epoch weights.")

    # Final test evaluation
    with torch.no_grad():
        test_loss, test_metrics, test_fairness = run_epoch(
            model, test_loader, device, train=False, optimizer=None
        )

    print("===== FINAL TEST RESULTS =====")
    print(
        f"[Test] loss={test_loss:.4f}, "
        f"AUC={test_metrics['auc']:.4f}, ACC={test_metrics['acc']:.4f}, F1={test_metrics['f1']:.4f}"
    )
    print(
        f"       EO(low-high)={test_fairness.get('eo_low_high', float('nan')):.4f}, "
        f"ACC var={test_fairness.get('acc_var', float('nan')):.6f}"
    )


if __name__ == "__main__":
    main()
