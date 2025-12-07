import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# Ensure we can import from dkt folder
# Assuming directory structure:
# root/
#   dkt/
#   gkt/
#     dkt_generator.py
#     train.py

# Add root/dkt to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
dkt_dir = os.path.join(root_dir, 'dkt')

if dkt_dir not in sys.path:
    sys.path.append(dkt_dir)

try:
    from dkt import DKT
except ImportError:
    # Fallback: try adding root to path
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    try:
        from dkt.dkt import DKT
    except ImportError:
        print("Warning: Could not import DKT model definition. Ensure 'dkt' folder is in the workspace root.")

def load_dkt_model(model_path, device):
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'args' in checkpoint and 'meta' in checkpoint:
        args = checkpoint['args']
        meta = checkpoint['meta']
        model_state = checkpoint['model_state']
    else:
        raise ValueError("Checkpoint does not contain 'args' and 'meta'. Ensure it was saved by train_dkt_merged.py")
    
    num_questions = int(meta['num_questions'])
    hidden_dim = int(args.get('hidden', 200))
    dropout = float(args.get('dropout', 0.1))
    
    print(f"Initializing DKT(num_c={num_questions}, emb_size={hidden_dim})")
    model = DKT(num_c=num_questions, emb_size=hidden_dim, dropout=dropout, emb_type='qid')
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model, num_questions

def get_gkt_permutation(gkt_csv_path, dkt_mapping_path):
    """
    Returns a list `perm` where perm[gkt_idx] = dkt_idx.
    """
    print(f"Loading GKT data from {gkt_csv_path}...")
    df = pd.read_csv(gkt_csv_path)
    
    if 'skill_id' not in df.columns:
        raise ValueError("merged_gkt.csv must contain 'skill_id' column")
        
    unique_skills = sorted(df['skill_id'].unique())
    gkt_idx_to_skill_id = {i: skill for i, skill in enumerate(unique_skills)}
    print(f"Found {len(unique_skills)} unique skills in GKT data.")

    print(f"Loading DKT mapping from {dkt_mapping_path}...")
    with open(dkt_mapping_path, 'r') as f:
        dkt_idx_to_qid = json.load(f)
    
    qid_to_dkt_idx = {str(v).strip(): int(k) for k, v in dkt_idx_to_qid.items()}
    
    perm = []
    missing_count = 0
    
    for i in range(len(unique_skills)):
        skill_id = gkt_idx_to_skill_id[i]
        qid_str = str(skill_id).strip()
        
        if qid_str in qid_to_dkt_idx:
            perm.append(qid_to_dkt_idx[qid_str])
        else:
            found = False
            try:
                if qid_str.endswith(".0"):
                    qid_str_alt = qid_str[:-2]
                    if qid_str_alt in qid_to_dkt_idx:
                        perm.append(qid_to_dkt_idx[qid_str_alt])
                        found = True
                elif "." not in qid_str:
                     qid_str_alt = qid_str + ".0"
                     if qid_str_alt in qid_to_dkt_idx:
                        perm.append(qid_to_dkt_idx[qid_str_alt])
                        found = True
            except:
                pass
            
            if not found:
                perm.append(-1)
                missing_count += 1
            
    if missing_count > 0:
        print(f"Warning: {missing_count} GKT skills could not be mapped to DKT indices. These rows/cols will be zeroed.")
        
    return perm

def generate_graph(model, num_questions, device):
    print("Generating DKT graph...")
    q_input = torch.arange(num_questions, device=device).unsqueeze(1)
    r_input = torch.ones(num_questions, 1, device=device)
    
    with torch.no_grad():
        y = model(q_input, r_input)
        
    adj = y.squeeze(1).cpu().numpy()
    return adj

def generate_and_save_dkt_graph(dkt_model_path, dkt_mapping_path, gkt_csv_path, output_path, device_str='cpu'):
    """
    Main entry point called by train.py
    """
    device = torch.device(device_str)
    
    # 1. Load DKT Model
    model, num_dkt_questions = load_dkt_model(dkt_model_path, device)
    
    # 2. Generate Raw DKT Graph
    raw_adj = generate_graph(model, num_dkt_questions, device)
    
    # 3. Determine Permutation
    perm = get_gkt_permutation(gkt_csv_path, dkt_mapping_path)
    
    num_gkt_questions = len(perm)
    print(f"Constructing GKT-aligned graph ({num_gkt_questions}x{num_gkt_questions})...")
    
    # 4. Construct GKT Graph
    gkt_adj = np.zeros((num_gkt_questions, num_gkt_questions), dtype=np.float32)
    
    for u in range(num_gkt_questions):
        dkt_u = perm[u]
        if dkt_u == -1:
            continue
        for v in range(num_gkt_questions):
            dkt_v = perm[v]
            if dkt_v == -1:
                continue
            gkt_adj[u, v] = raw_adj[dkt_u, dkt_v]
            
    # 5. Save
    print(f"Saving graph to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savetxt(output_path, gkt_adj, fmt='%.6f')
    print("DKT Graph generation complete.")
