import pandas as pd
import numpy as np
import json
import os
import argparse

def generate_splits(csv_path, output_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Normalize User ID column
    if "UserId" in df.columns:
        df.rename(columns={"UserId": "user_id"}, inplace=True)
    
    # Get unique users
    users = df["user_id"].unique()
    users = sorted([str(u) for u in users]) # Sort to ensure deterministic order before shuffle
    n_users = len(users)
    print(f"Found {n_users} unique users.")
    
    # Shuffle with fixed seed
    rng = np.random.default_rng(seed)
    indices = np.arange(n_users)
    rng.shuffle(indices)
    
    # Calculate split points
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    
    splits = {
        "train": [users[i] for i in train_idx],
        "valid": [users[i] for i in val_idx],
        "test": [users[i] for i in test_idx]
    }
    
    print(f"Split sizes: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Saved unified splits to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="merged_student_question_history.csv")
    parser.add_argument("--output", default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_splits(args.input, args.output, seed=args.seed)
