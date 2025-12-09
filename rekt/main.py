import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import ReKT, ReKT_concept
from run import run_epoch

# Path configuration for the MERGED dataset produced by merged_data.py
mp2path = {
    "merged": {
        "ques_skill_path": "data/MERGED/ques_skill.csv",
        "train_path": "data/MERGED/train_question.txt",
        "valid_path": "data/MERGED/valid_question.txt",
        "test_path": "data/MERGED/test_question.txt",
        "train_skill_path": "data/MERGED/train_skill.txt",
        "valid_skill_path": "data/MERGED/valid_skill.txt",
        "test_skill_path": "data/MERGED/test_skill.txt",
        # No fixed skill_max here; we infer it from ques_skill.csv
    }
}


def get_problem_and_skill_max(ques_skill_path: str):
    """Infer pro_max and skill_max from ques_skill.csv.

    ques_skill.csv is expected to have at least two columns:
        question_idx, skill_idx

    We use:
        pro_max   = 1 + max(question_idx)
        skill_max = 1 + max(skill_idx)
    """  # noqa: D401
    df = pd.read_csv(ques_skill_path)
    if df.shape[1] < 2:
        raise ValueError(
            "ques_skill.csv must have at least two columns "
            "(question_idx, skill_idx)."
        )
    q_max = int(df.iloc[:, 0].max())
    s_max = int(df.iloc[:, 1].max())
    pro_max = q_max + 1
    skill_max = s_max + 1
    return pro_max, skill_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReKT model")
    parser.add_argument("--dataset", type=str, default="merged", help="Dataset name")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension (d)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (p)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs with different seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (if num_runs=1)")
    
    args = parser.parse_args()

    # You can change this to run other datasets if you extend mp2path
    dataset = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths = mp2path[dataset]

    ques_skill_path = paths["ques_skill_path"]
    train_path = paths["train_path"]
    valid_path = paths["valid_path"]
    test_path = paths["test_path"]
    train_skill_path = paths["train_skill_path"]
    valid_skill_path = paths["valid_skill_path"]
    test_skill_path = paths["test_skill_path"]

    # Infer pro_max and skill_max from ques_skill.csv
    pro_max, skill_max = get_problem_and_skill_max(ques_skill_path)
    print(f"[{dataset}] pro_max = {pro_max}, skill_max = {skill_max}")

    # Hyperparameters
    p = args.dropout
    d = args.hidden
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    min_seq = 3
    max_seq = 310
    grad_clip = 15.0
    patience = 15  # early-stopping patience

    avg_auc = 0.0
    avg_acc = 0.0
    avg_f1 = 0.0

    # For reproducibility / robustness, do num_runs with different seeds
    num_runs = args.num_runs

    with open(f"{dataset}_output.txt", "w") as file:
        for run_id in range(num_runs):
            print("====================================================================")
            print(f"Run {run_id + 1}/{num_runs} for dataset = {dataset}")
            print("====================================================================")

            # Use args.seed if num_runs is 1, else use run_id
            current_seed = args.seed if num_runs == 1 else run_id
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

            best_acc = 0.0
            best_auc = 0.0
            best_f1 = 0.0
            best_state = {"auc": 0.0, "acc": 0.0, "f1": 0.0, "loss": 0.0}

            model = ReKT(pro_max, skill_max, d, p)
            model = model.to(device)

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=1e-5
            )

            epochs_no_improve = 0

            for epoch in range(epochs):
                # Train
                train_loss, train_acc, train_auc, train_f1 = run_epoch(
                    pro_max,
                    train_path,
                    train_skill_path,
                    batch_size,
                    True,
                    min_seq,
                    max_seq,
                    model,
                    optimizer,
                    criterion,
                    device,
                    grad_clip,
                )
                print(
                    f"epoch: {epoch:03d}, "
                    f"train_loss: {train_loss:.4f}, "
                    f"train_acc: {train_acc:.4f}, "
                    f"train_auc: {train_auc:.4f}, "
                    f"train_f1: {train_f1:.4f}"
                )

                # Validation: here we reuse the test set as 'valid' set.
                # If you later create a distinct validation split, point
                # valid_path/valid_skill_path there instead.
                valid_loss, valid_acc, valid_auc, valid_f1 = run_epoch(
                    pro_max,
                    valid_path,
                    valid_skill_path,
                    batch_size,
                    False,
                    min_seq,
                    max_seq,
                    model,
                    optimizer,
                    criterion,
                    device,
                    grad_clip,
                    prefix="Validation",
                )
                print(
                    f"epoch: {epoch:03d}, "
                    f"valid_loss: {valid_loss:.4f}, "
                    f"valid_acc: {valid_acc:.4f}, "
                    f"valid_auc: {valid_auc:.4f}, "
                    f"valid_f1: {valid_f1:.4f}"
                )

                # Early stopping on AUC
                if valid_auc >= best_auc:
                    best_auc = valid_auc
                    best_acc = valid_acc
                    best_f1 = valid_f1
                    best_state["auc"] = valid_auc
                    best_state["acc"] = valid_acc
                    best_state["f1"] = valid_f1
                    best_state["loss"] = valid_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch}, "
                        f"best_valid_auc={best_auc:.4f}, "
                        f"best_valid_acc={best_acc:.4f}, "
                        f"best_valid_f1={best_f1:.4f}"
                    )
                    break

            # After training, evaluate once more on test set
            test_loss, test_acc, test_auc, test_f1 = run_epoch(
                pro_max,
                test_path,
                test_skill_path,
                batch_size,
                False,
                min_seq,
                max_seq,
                model,
                optimizer,
                criterion,
                device,
                grad_clip,
                prefix="Test",
            )

            print("************************************************************************")
            print(f"Run {run_id + 1}/{num_runs} TEST:")
            print(f"test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}, test_f1: {test_f1:.4f}")
            print("************************************************************************")

            avg_auc += test_auc
            avg_acc += test_acc
            avg_f1 += test_f1

            # Write per-run summary to file
            file.write(
                f"Run {run_id + 1}: "
                f"best_valid_auc={best_auc:.4f}, "
                f"best_valid_acc={best_acc:.4f}, "
                f"best_valid_f1={best_f1:.4f}, "
                f"test_auc={test_auc:.4f}, "
                f"test_acc={test_acc:.4f}, "
                f"test_f1={test_f1:.4f}\n"
            )

        avg_auc /= num_runs
        avg_acc /= num_runs
        avg_f1 /= num_runs

        print("====================================================================")
        print(f"FINAL AVERAGE over {num_runs} runs - dataset: {dataset}")
        print(f"final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}, final_avg_f1: {avg_f1:.4f}")
        print("====================================================================")

        file.write(
            f"FINAL AVERAGE over {num_runs} runs: "
            f"acc={avg_acc:.4f}, auc={avg_auc:.4f}, f1={avg_f1:.4f}\n"
        )
