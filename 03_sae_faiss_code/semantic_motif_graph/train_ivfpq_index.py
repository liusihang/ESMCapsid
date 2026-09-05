#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an IVFPQ index on CPU using deterministic blockwise sampling and automatic padding.
Usage:
python train_ivfpq_cpu.py --input sae_feats.npz --output_index ivfpq_trained.index \
    --metric cosine --nlist 65536 --m 128 --nbits 8 --train_size 1000000 \
    --block_size 1000000 --seed 42
"""

import argparse
import numpy as np
from scipy.sparse import csr_matrix, vstack
import faiss
import math

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, nargs='+', required=True)
    p.add_argument('--output_index', type=str, required=True)
    p.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2', 'ip'])
    p.add_argument('--nlist', type=int, default=65536)
    p.add_argument('--m', type=int, default=128)
    p.add_argument('--nbits', type=int, default=8)
    p.add_argument('--train_size', type=int, default=1000000)
    p.add_argument('--block_size', type=int, default=1000000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def load_csr_npz(npz_file):
    data_npz = np.load(npz_file)
    required_keys = ['data', 'indices', 'indptr', 'shape']
    for k in required_keys:
        if k not in data_npz:
            raise ValueError(f"NPZ is missing field: {k}")
    return csr_matrix(
        (data_npz['data'], data_npz['indices'], data_npz['indptr']),
        shape=tuple(data_npz['shape'])
    )

def load_multiple_csr(npz_files):
    mats = [load_csr_npz(f) for f in npz_files]
    if len(mats) == 1:
        return mats[0]
    return vstack(mats, format='csr')

def compute_padded_dim(D, m):
    if D % m == 0:
        return D
    return ((D + m - 1) // m) * m

def pad_dense(X, D_pad):
    if X.shape[1] == D_pad:
        return X
    X_pad = np.zeros((X.shape[0], D_pad), dtype=X.dtype)
    X_pad[:, :X.shape[1]] = X
    return X_pad

def blockwise_random_sample_indices(n_samples, train_size, block_size, seed):
    """
    Deterministically allocate samples across blocks so the total equals train_size.
    """
    rng = np.random.default_rng(seed)
    indices = []

    remaining = train_size
    start = 0
    while start < n_samples:
        end = min(start + block_size, n_samples)
        block_n = end - start
        remaining_rows = n_samples - start

        if end == n_samples:
            sample_n = remaining
        else:
            
            sample_n = int(round(remaining * block_n / remaining_rows))
            sample_n = min(sample_n, block_n)

        if sample_n > 0:
            local_idx = rng.choice(block_n, size=sample_n, replace=False)
            indices.append(local_idx + start)
            remaining -= sample_n

        start = end
        if remaining <= 0:
            break

    idx = np.concatenate(indices)
    if idx.size != train_size:
        raise RuntimeError(f"Sample count mismatch: {idx.size} != {train_size}")
    return idx

def main():
    args = parse_args()
    X = load_multiple_csr(args.input)
    n_samples, D = X.shape

    if args.train_size <= 0:
        raise ValueError("train_size must be greater than zero")
    if args.train_size > n_samples:
        args.train_size = n_samples

    if args.train_size < args.nlist:
        raise ValueError(f"train_size={args.train_size} is too small and must be at least nlist={args.nlist}")

    
    D_pad = compute_padded_dim(D, args.m)
    print(f"Original dimension D={D}, padded dimension D_pad={D_pad} (m={args.m})")

    
    idx = blockwise_random_sample_indices(n_samples, args.train_size, args.block_size, args.seed)
    idx.sort()  
    train_x = X[idx].toarray().astype(np.float32)
    train_x = pad_dense(train_x, D_pad)

    
    if args.metric == 'cosine':
        faiss.normalize_L2(train_x)

    
    if args.metric in ['cosine', 'ip']:
        quantizer = faiss.IndexFlatIP(D_pad)
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        quantizer = faiss.IndexFlatL2(D_pad)
        metric = faiss.METRIC_L2

    index = faiss.IndexIVFPQ(quantizer, D_pad, args.nlist, args.m, args.nbits, metric)
    print(f"[CPU] training IVFPQ: nlist={args.nlist}, m={args.m}, nbits={args.nbits}")
    index.train(train_x)

    faiss.write_index(index, args.output_index)
    print(f" Training complete; saved: {args.output_index}")

if __name__ == "__main__":
    main()