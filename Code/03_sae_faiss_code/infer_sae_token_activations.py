#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory-efficient SAE token-activation inference."""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import shutil


# ==========================================

# ==========================================
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_dict, topk):
        super().__init__()
        self.d_in = d_in
        self.d_dict = d_dict
        self.topk = int(topk)

        self.W_enc = nn.Parameter(torch.empty(d_in, d_dict))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.register_buffer("b_pre", torch.zeros(d_in))
        self.register_buffer("alpha", torch.tensor(1.0))
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_in))
        
        # self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _center_scale(self, x):
        return (x - self.b_pre) * self.alpha

    def encode(self, x):
        x_cs = self._center_scale(x)
        pre_acts = x_cs @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        topk_vals, topk_idx = torch.topk(acts, self.topk, dim=1)
        return topk_idx, topk_vals


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_global_stats(output_root):
    stats_path = os.path.join(output_root, "global_stats.npz")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(os.path.dirname(output_root), "global_stats.npz")
    return np.load(stats_path) if os.path.exists(stats_path) else None


# ==========================================

# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--sequences", type=int, default=2000000)
    parser.add_argument(
        "--batch_size", type=int, default=50240
    )  
    args = parser.parse_args()

    run_dir = args.run_dir
    target_seqs = args.sequences

    
    folder_name = os.path.basename(os.path.normpath(run_dir))
    parts = folder_name.split("_")
    d_dict = int([p for p in parts if p.startswith("D")][0][1:])
    k = int([p for p in parts if p.startswith("K")][0][1:])

    pre_cfg_path = os.path.join(run_dir, "preprocess_config.json")
    if not os.path.exists(pre_cfg_path):
        pre_cfg_path = os.path.join(os.path.dirname(run_dir), "preprocess_config.json")
    with open(pre_cfg_path) as f:
        pre_config = json.load(f)

    input_tokens_path = pre_config.get("input_tokens_path")
    if not os.path.exists(input_tokens_path):
        input_tokens_path = os.path.join(
            os.path.dirname(run_dir), os.path.basename(input_tokens_path)
        )

    X_mmap = np.load(input_tokens_path, mmap_mode="r")
    offsets = np.load(
        os.path.join(os.path.dirname(input_tokens_path), "seq_token_offsets.npy")
    )

    
    total_avail_seqs = len(offsets)
    target_seqs = min(target_seqs, total_avail_seqs)
    np.random.seed(42)
    selected_seq_ids = np.sort(
        np.random.choice(total_avail_seqs, target_seqs, replace=False)
    )

    
    ranges = []
    total_tokens = 0
    for seq_id in selected_seq_ids:
        start = offsets[seq_id]
        end = offsets[seq_id + 1] if seq_id < total_avail_seqs - 1 else X_mmap.shape[0]
        if end > start:
            ranges.append((start, end, seq_id))
            total_tokens += end - start

    out_prefix = os.path.join(run_dir, f"sae_feats_sampled_{target_seqs}seqs")
    temp_dir = os.path.join(run_dir, "temp_mmap")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    
    
    meta_csv_path = f"{out_prefix}_meta.csv"
    with open(meta_csv_path, "w") as f_meta:
        f_meta.write("Token_ID,Seq_ID,Position\n")  

    
    total_nnz = total_tokens * k
    mmap_rows = np.memmap(
        os.path.join(temp_dir, "rows.bin"), dtype="int64", mode="w+", shape=(total_nnz,)
    )
    mmap_cols = np.memmap(
        os.path.join(temp_dir, "cols.bin"), dtype="int32", mode="w+", shape=(total_nnz,)
    )
    mmap_vals = np.memmap(
        os.path.join(temp_dir, "vals.bin"),
        dtype="float32",
        mode="w+",
        shape=(total_nnz,),
    )

    
    device = get_device()
    sae = TopKSAE(pre_config["d_in"], d_dict, k).to(device)
    
    sae.load_state_dict(
        torch.load(os.path.join(run_dir, "sae_model.pt"), map_location=device),
        strict=False,
    )
    stats = load_global_stats(os.path.dirname(run_dir))
    if stats is not None:
        sae.b_pre.copy_(torch.from_numpy(stats["mu"]).to(device))
        sae.alpha.fill_(float(stats["alpha"]))
    sae.eval()

    
    curr_nnz_idx = 0
    curr_token_offset = 0

    batch_tokens = []
    batch_meta = []  

    def flush_batch():
        nonlocal curr_nnz_idx, curr_token_offset, batch_tokens, batch_meta
        nonlocal mmap_rows, mmap_cols, mmap_vals
        if not batch_tokens:
            return

        n_samples = len(batch_tokens)
        inp = torch.from_numpy(np.stack(batch_tokens)).float().to(device)

        with torch.no_grad():
            topk_idx, topk_vals = sae.encode(inp)

        
        n_elements = n_samples * k
        rows = np.repeat(
            np.arange(curr_token_offset, curr_token_offset + n_samples, dtype=np.int64),
            k,
        )

        mmap_rows[curr_nnz_idx : curr_nnz_idx + n_elements] = rows
        mmap_cols[curr_nnz_idx : curr_nnz_idx + n_elements] = (
            topk_idx.cpu().numpy().flatten()
        )
        mmap_vals[curr_nnz_idx : curr_nnz_idx + n_elements] = (
            topk_vals.cpu().numpy().flatten()
        )

        
        df_m = pd.DataFrame(batch_meta)
        df_m.to_csv(meta_csv_path, mode="a", index=False, header=False)

        
        curr_nnz_idx += n_elements
        curr_token_offset += n_samples
        batch_tokens, batch_meta = [], []

    print(f"Processing {total_tokens} tokens...")
    pbar = tqdm(total=total_tokens, unit="tok")

    for start, end, seq_id in ranges:
        seq_data = X_mmap[start:end]
        seq_len = seq_data.shape[0]

        for i in range(seq_len):
            batch_tokens.append(seq_data[i])
            batch_meta.append({"Token_ID": start + i, "Seq_ID": seq_id, "Position": i})

            if len(batch_tokens) >= args.batch_size:
                flush_batch()
                pbar.update(args.batch_size)

    if batch_tokens:
        pbar.update(len(batch_tokens))
        flush_batch()
    pbar.close()

    
    print("Constructing final CSR matrix (this may take a moment)...")
    
    mmap_rows.flush()
    mmap_cols.flush()
    mmap_vals.flush()

    
    X_csr = sp.csr_matrix(
        (mmap_vals, (mmap_rows, mmap_cols)),
        shape=(total_tokens, d_dict),
        dtype=np.float32,
    )

    print(f"Saving to {out_prefix}.npz ...")
    sp.save_npz(f"{out_prefix}.npz", X_csr)

    
    del mmap_rows, mmap_cols, mmap_vals  
    shutil.rmtree(temp_dir)
    print("Done. All outputs saved.")


if __name__ == "__main__":
    main()
