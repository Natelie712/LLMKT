
from sklearn import metrics
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
import numpy as np
import math

from load_data import getLoader


def _compute_bin_stats(bin_labels, bin_outputs):
    """Compute per-bin TPR, FPR, ACC for 10 completion bins.

    Parameters
    ----------
    bin_labels : list[list[int]]
        For each bin 0..9, a list of 0/1 labels.
    bin_outputs : list[list[float]]
        For each bin 0..9, a list of predicted probabilities.

    Returns
    -------
    dict
        Mapping bin_index -> dict(tpr=..., fpr=..., acc=..., count=...),
        only for bins that have at least one example.
    """
    stats = {}
    for b in range(10):
        y_true = np.asarray(bin_labels[b], dtype=int)
        y_score = np.asarray(bin_outputs[b], dtype=float)

        if y_true.size == 0:
            continue

        y_pred = (y_score >= 0.5).astype(int)

        pos_mask = y_true == 1
        neg_mask = y_true == 0

        tp = int((y_pred[pos_mask] == 1).sum()) if pos_mask.any() else 0
        fn = int((y_pred[pos_mask] == 0).sum()) if pos_mask.any() else 0
        fp = int((y_pred[neg_mask] == 1).sum()) if neg_mask.any() else 0
        tn = int((y_pred[neg_mask] == 0).sum()) if neg_mask.any() else 0

        tpr = float(tp) / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = float(fp) / (fp + tn) if (fp + tn) > 0 else float("nan")
        denom = tp + tn + fp + fn
        acc = float(tp + tn) / denom if denom > 0 else float("nan")

        stats[b] = {"tpr": tpr, "fpr": fpr, "acc": acc, "count": y_true.size}

    return stats


def _compute_fairness_from_bins(bin_stats):
    """Compute Euclidean equalized odds distance and accuracy variance.

    Parameters
    ----------
    bin_stats : dict
        Output of _compute_bin_stats.

    Returns
    -------
    dict
        {
          "eo_dist_low_high": float or nan,
          "acc_var": float or nan,
        }
    """
    if not bin_stats:
        return {"eo_dist_low_high": float("nan"), "acc_var": float("nan")}

    # sort non-empty bins
    non_empty_bins = sorted(bin_stats.keys())
    low = non_empty_bins[0]
    high = non_empty_bins[-1]

    low_stat = bin_stats[low]
    high_stat = bin_stats[high]

    tpr_low, fpr_low = low_stat["tpr"], low_stat["fpr"]
    tpr_high, fpr_high = high_stat["tpr"], high_stat["fpr"]

    if (
        np.isnan(tpr_low)
        or np.isnan(fpr_low)
        or np.isnan(tpr_high)
        or np.isnan(fpr_high)
    ):
        eo_dist = float("nan")
    else:
        # Euclidean distance in (TPR, FPR) space
        eo_dist = math.sqrt((tpr_low - tpr_high) ** 2 + (fpr_low - fpr_high) ** 2)

    # Accuracy variance across all non-empty bins
    accs = [
        st["acc"]
        for st in bin_stats.values()
        if not np.isnan(st["acc"])
    ]
    acc_var = np.var(accs) if accs else float("nan")

    return {"eo_dist_low_high": eo_dist, "acc_var": acc_var}


def run_epoch(
    max_problem,
    pro_path,
    skill_path,
    batch_size,
    is_train,
    min_problem_num,
    max_problem_num,
    model,
    optimizer,
    criterion,
    device,
    grad_clip,
):
    """Run one training or evaluation epoch.

    This keeps the original ReKT API and metrics (loss, ACC, AUC) but,
    in evaluation mode (is_train == False), it additionally computes:

      * completion rate per student: L / Q
          - L = number of valid timesteps (sum of mask)
          - Q = maximum possible sequence length (len over time dimension)
      * 10 completion bins: [0,10%), [10,20%), ..., [90,100%]
      * per-bin TPR, FPR, ACC
      * Euclidean equalized-odds distance between lowest and highest
        non-empty completion bins
      * accuracy variance across all non-empty bins

    Fairness statistics are *only* logged during evaluation. The return
    value is (avg_loss, acc, auc, f1).
    """  # noqa: E501
    loader = getLoader(
        max_problem, pro_path, skill_path, batch_size, is_train, min_problem_num, max_problem_num
    )

    if is_train:
        model.train()
    else:
        model.eval()

    total_correct = 0
    total_num = 0
    total_loss = []

    labels = []
    outputs = []

    # For fairness metrics (evaluation only)
    if not is_train:
        # 10 completion bins: 0..9
        bin_stats = [{"labels": [], "outputs": [], "users": set()} for _ in range(10)]

    for batch in tqdm(loader):
        # Unpack batch (now includes completion_rate and user_id per sample)
        last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, mask, completion_rate, user_id = batch

        # Everything should already be on device from load_data, but this is safe
        last_problem = last_problem.to(device)
        last_skill = last_skill.to(device)
        last_ans = last_ans.to(device)
        next_problem = next_problem.to(device)
        next_skill = next_skill.to(device)
        next_ans = next_ans.to(device).float()
        mask = mask.to(device)

        if is_train:
            optimizer.zero_grad()

            next_predict = model(
                last_problem, last_skill, last_ans, next_problem, next_skill, next_ans
            )  # [B, L]

            # Only compute loss/metrics on valid positions
            valid_pred = next_predict[mask]
            valid_true = next_ans[mask]

            loss = criterion(valid_pred, valid_true)
            loss.backward()

            # Gradient clipping
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
        else:
            with torch.no_grad():
                next_predict = model(
                    last_problem, last_skill, last_ans, next_problem, next_skill, next_ans
                )  # [B, L]

                valid_pred = next_predict[mask]
                valid_true = next_ans[mask]

                loss = criterion(valid_pred, valid_true)

        # Collect global metrics (both train and eval)
        labels.extend(valid_true.view(-1).detach().cpu().numpy())
        outputs.extend(valid_pred.view(-1).detach().cpu().numpy())

        total_loss.append(loss.item())
        total_num += len(valid_true.view(-1))

        to_pred = (valid_pred >= 0.5).long()
        total_correct += (valid_true.long() == to_pred).sum().item()

        # Fairness bookkeeping: per-student completion bins (eval only)
        if not is_train:
            # mask/next_predict/next_ans have shape [B, L]
            batch_size_cur = next_ans.shape[0]

            # For each sample in batch, assign all valid steps to a bin
            # based on precomputed completion rate
            for i in range(batch_size_cur):
                # completion_rate is already calculated as: student's total questions / 305
                cr = float(completion_rate[i].item() if torch.is_tensor(completion_rate[i]) else completion_rate[i])
                uid = int(user_id[i].item() if torch.is_tensor(user_id[i]) else user_id[i])
                
                # Map completion_rate in [0,1] to bins 0..9
                if cr >= 1.0:
                    bin_idx = 9
                elif cr <= 0.0:
                    bin_idx = 0
                else:
                    bin_idx = int(cr * 10.0)
                
                # Clamp to valid range
                bin_idx = max(0, min(9, bin_idx))
                
                # Track unique user in this bin
                bin_stats[bin_idx]["users"].add(uid)
                
                # Get valid positions for this sample
                m_i = mask[i].bool()
                if m_i.sum() == 0:
                    continue

                student_labels = next_ans[i][m_i].detach().cpu().numpy().astype(int)
                student_outputs = next_predict[i][m_i].detach().cpu().numpy().astype(float)

                bin_stats[bin_idx]["labels"].extend(student_labels.tolist())
                bin_stats[bin_idx]["outputs"].extend(student_outputs.tolist())

    avg_loss = float(np.average(total_loss)) if total_loss else float("nan")
    acc = float(total_correct) / float(total_num) if total_num > 0 else float("nan")

    # Guard against AUC failure if labels are all one class
    try:
        auc = metrics.roc_auc_score(labels, outputs)
    except ValueError:
        auc = float("nan")
    
    # Calculate F1 score
    try:
        preds = (np.array(outputs) >= 0.5).astype(int)
        f1 = f1_score(labels, preds)
    except (ValueError, ZeroDivisionError):
        f1 = float("nan")

    # Fairness metrics (evaluation only)
    if not is_train:
        # Compute metrics from collected bin statistics
        computed_bin_stats = {}
        for b in range(10):
            y_true = np.asarray(bin_stats[b]["labels"], dtype=int)
            y_score = np.asarray(bin_stats[b]["outputs"], dtype=float)
            num_students = len(bin_stats[b]["users"])
            
            if y_true.size == 0:
                continue
                
            y_pred = (y_score >= 0.5).astype(int)
            
            pos_mask = y_true == 1
            neg_mask = y_true == 0
            
            tp = int((y_pred[pos_mask] == 1).sum()) if pos_mask.any() else 0
            fn = int((y_pred[pos_mask] == 0).sum()) if pos_mask.any() else 0
            fp = int((y_pred[neg_mask] == 1).sum()) if neg_mask.any() else 0
            tn = int((y_pred[neg_mask] == 0).sum()) if neg_mask.any() else 0
            
            tpr = float(tp) / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = float(fp) / (fp + tn) if (fp + tn) > 0 else float("nan")
            denom = tp + tn + fp + fn
            acc = float(tp + tn) / denom if denom > 0 else float("nan")
            
            computed_bin_stats[b] = {
                "tpr": tpr,
                "fpr": fpr,
                "acc": acc,
                "students": num_students,
                "predictions": y_true.size
            }
        
        fairness = _compute_fairness_from_bins(computed_bin_stats)

        # Pretty-print fairness summary
        print("Fairness per completion-rate bin:")
        for b in sorted(computed_bin_stats.keys()):
            st = computed_bin_stats[b]
            print(
                f"  Bin {b} ({b*10:2d}-{(b+1)*10:3d}%): "
                f"students={st['students']}, "
                f"predictions={st['predictions']}, "
                f"TPR={st['tpr']:.3f}, "
                f"FPR={st['fpr']:.3f}, "
                f"ACC={st['acc']:.3f}"
            )
        print(
            f"Equalized odds distance (lowest vs highest non-empty bin): "
            f"{fairness['eo_dist_low_high']:.4f}"
        )
        print(f"Accuracy variance across bins: {fairness['acc_var']:.6f}")

    return avg_loss, acc, auc, f1
