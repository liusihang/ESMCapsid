#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a trained IVFPQ index, add vectors and search k-nearest neighbours on GPU.
Usage:
python search_ivfpq_gpu.py --input sae_feats.npz --trained_index ivfpq_trained.index \
    --output knn_results.npz --k 64 --metric cosine --nprobe 64 --gpu 0
"""

import argparse
import numpy as np
from scipy.sparse import csr_matrix, vstack
import faiss
import os
import time
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, nargs='+', required=True)
    p.add_argument('--trained_index', type=str, required=True)
    p.add_argument('--output', type=str, default='knn_results.npz')
    p.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'l2', 'ip'])
    p.add_argument('--k', type=int, default=64)
    p.add_argument('--nprobe', type=int, default=64)
    p.add_argument('--gpu', type=int, default=0, help='-1: all GPUs, -2: CPU')
    p.add_argument('--add_batch_size', type=int, default=200000)
    p.add_argument('--query_batch_size', type=int, default=100000)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--use_float16', action='store_true', help='Use FP16 for the GPU index')
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

def pad_dense(X, D_pad):
    if X.shape[1] == D_pad:
        return X
    X_pad = np.zeros((X.shape[0], D_pad), dtype=X.dtype)
    X_pad[:, :X.shape[1]] = X
    return X_pad

def main():
    args = parse_args()

    X = load_multiple_csr(args.input)
    if args.max_samples and X.shape[0] > args.max_samples:
        X = X[:args.max_samples]
    n_samples, D = X.shape
    print(f"Data shape: {n_samples:,}  {D:,}")

    
    index_cpu = faiss.read_index(args.trained_index)
    D_index = index_cpu.d
    if D_index < D:
        raise ValueError(f"Trained index dimension D_index={D_index} < data dimension D={D}")

    
    expected = faiss.METRIC_INNER_PRODUCT if args.metric in ['cosine', 'ip'] else faiss.METRIC_L2
    if index_cpu.metric_type != expected:
        raise ValueError("The requested metric does not match the trained index.")

    
    if args.gpu >= 0:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = args.use_float16
        co.useFloat16CoarseQuantizer = args.use_float16
        index = faiss.index_cpu_to_gpu(res, args.gpu, index_cpu, co)
        print(f" using GPU {args.gpu}, FP16={args.use_float16}")
    elif args.gpu == -1:
        index = faiss.index_cpu_to_all_gpus(index_cpu)
        print(" using all GPUs")
    else:
        index = index_cpu
        print(" using CPU")

    index.nprobe = args.nprobe

    
    print("Adding vectors to the index...")
    for i in tqdm(range(0, n_samples, args.add_batch_size), desc="add index vectors"):
        end = min(i + args.add_batch_size, n_samples)
        batch = X[i:end].toarray().astype(np.float32)
        batch = pad_dense(batch, D_index)
        if args.metric == 'cosine':
            faiss.normalize_L2(batch)
        index.add(batch)

    print(f"Index vector count: {index.ntotal:,}")

    
    all_dist = np.zeros((n_samples, args.k), dtype=np.float32)
    all_idx = np.zeros((n_samples, args.k), dtype=np.int64)

    print("Starting kNN search...")
    t0 = time.time()
    for i in tqdm(range(0, n_samples, args.query_batch_size), desc="kNN search"):
        end = min(i + args.query_batch_size, n_samples)
        query = X[i:end].toarray().astype(np.float32)
        query = pad_dense(query, D_index)
        if args.metric == 'cosine':
            faiss.normalize_L2(query)

        D_batch, I_batch = index.search(query, args.k)
        all_dist[i:end] = D_batch
        all_idx[i:end] = I_batch

    elapsed = time.time() - t0
    print(f" Search complete; elapsed {elapsed/60:.2f} minutes")

    np.savez_compressed(
        args.output,
        neighbors=all_idx,
        distances=all_dist,
        k=args.k,
        n_samples=n_samples,
        D=D,
        D_index=D_index,
        metric=args.metric,
        nprobe=args.nprobe
    )
    size_gb = os.path.getsize(args.output) / 1024**3
    print(f"Results saved: {args.output} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    main()