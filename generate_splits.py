import pandas as pd
import numpy as np
import json
import os
import argparse

def generate_splits(csv_path, output_path, train_ratio=0.7, val_ratio=0.1, seed=42, stratify_by_completion=False):
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
    
    if stratify_by_completion:
        print("Using stratified splitting by completion rate...")
        # Calculate completion rate for each user (efficient version)
        user_completion = {}
        tot_q_slots = 305
        
        # Get all question ID columns
        qid_cols = [f"q{n}_id" for n in range(1, tot_q_slots + 1) if f"q{n}_id" in df.columns]
        
        # Count answered questions per user in one pass
        print(f"Calculating completion rates for {n_users} users...")
        for _, row in df.iterrows():
            user_id = str(row["user_id"])
            answered = sum(1 for col in qid_cols if not pd.isna(row[col]))
            completion_rate = answered / tot_q_slots
            user_completion[user_id] = completion_rate
        
        # Assign users to bins (0-9)
        user_bins = {}
        for user_id, cr in user_completion.items():
            bin_idx = min(int(cr * 10), 9)
            user_bins[user_id] = bin_idx
        
        # Group users by bin
        bins = {}
        for i in range(10):
            bins[i] = []
        for user_id in users:
            bin_idx = user_bins[user_id]
            bins[bin_idx].append(user_id)
        
        print("User distribution across completion bins:")
        for i in range(10):
            if len(bins[i]) > 0:
                print(f"  Bin {i} ({i*10}-{(i+1)*10}%): {len(bins[i])} users")
        
        # Stratified split: ensure each bin appears in all splits when possible
        rng = np.random.default_rng(seed)
        train_users = []
        val_users = []
        test_users = []
        
        for bin_idx in range(10):
            bin_users = bins[bin_idx]
            if len(bin_users) == 0:
                continue
            
            # Shuffle bin users
            bin_indices = np.arange(len(bin_users))
            rng.shuffle(bin_indices)
            
            # For small bins (1-3 users), ensure each split gets at least 1 if possible
            if len(bin_users) <= 3:
                if len(bin_users) == 1:
                    # Put single user in test (largest split)
                    test_users.append(bin_users[0])
                elif len(bin_users) == 2:
                    # Put 1 in validation, 1 in test
                    val_users.append(bin_users[bin_indices[0]])
                    test_users.append(bin_users[bin_indices[1]])
                else:  # len == 3
                    # Put 1 in each split
                    train_users.append(bin_users[bin_indices[0]])
                    val_users.append(bin_users[bin_indices[1]])
                    test_users.append(bin_users[bin_indices[2]])
            else:
                # For larger bins, use proportional splitting
                n_train = int(len(bin_users) * train_ratio)
                n_val = int(len(bin_users) * val_ratio)
                
                # Ensure at least 1 in each split
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                n_test = len(bin_users) - n_train - n_val
                n_test = max(1, n_test)
                
                # Adjust if totals don't match
                while n_train + n_val + n_test != len(bin_users):
                    if n_train + n_val + n_test < len(bin_users):
                        n_train += 1
                    else:
                        n_train -= 1
                
                train_users.extend([bin_users[i] for i in bin_indices[:n_train]])
                val_users.extend([bin_users[i] for i in bin_indices[n_train:n_train + n_val]])
                test_users.extend([bin_users[i] for i in bin_indices[n_train + n_val:]])
        
        splits = {
            "train": train_users,
            "valid": val_users,
            "test": test_users
        }
        
        print(f"Stratified split sizes: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")
    else:
        # Original random splitting
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
    parser.add_argument("--stratify", action="store_true", 
                        help="Use stratified splitting by completion rate to ensure all bins represented")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_splits(args.input, args.output, seed=args.seed, stratify_by_completion=args.stratify)
