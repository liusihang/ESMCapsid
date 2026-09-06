#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map query SAE token features to an existing Leiden cluster system.

Pipeline:
1) Load a trained IVFPQ index template.
2) Add reference SAE features into index.
3) Search kNN for query SAE features.
4) Vote neighbors' reference Leiden labels to produce mapped labels.

Input CSR NPZ format requirement:
- keys: data, indices, indptr, shape
"""

import os
import json
import time
import math
import argparse

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import faiss


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Map SAE token features to reference Leiden cluster IDs"
    )
    p.add_argument("--query_npz", required=True, help="Query SAE CSR NPZ")
    p.add_argument(
        "--ref_npz",
        required=True,
        help="Reference SAE CSR NPZ used by original cluster system",
    )
    p.add_argument(
        "--ref_labels",
        required=True,
        help="Reference token-level Leiden labels (.npy)",
    )
    p.add_argument(
        "--trained_index",
        required=True,
        help="Trained IVFPQ index template (.index)",
    )
    p.add_argument("--output_prefix", default=None, help="Output prefix path")

    p.add_argument("--k", type=int, default=20, help="kNN neighbors for voting")
    p.add_argument("--nprobe", type=int, default=64, help="FAISS nprobe")
    p.add_argument("--gpu", type=int, default=0, help="GPU id, -2 for CPU")
    p.add_argument("--use_float16", action="store_true", help="Use FP16 on GPU index")

    p.add_argument(
        "--add_batch_size", type=int, default=200000, help="Reference add batch size"
    )
    p.add_argument(
        "--query_batch_size", type=int, default=100000, help="Query search batch size"
    )
    p.add_argument(
        "--vote_mode",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Voting mode for cluster mapping",
    )

    p.add_argument(
        "--max_ref_tokens", type=int, default=None, help="Debug: cap ref tokens"
    )
    p.add_argument(
        "--max_query_tokens", type=int, default=None, help="Debug: cap query tokens"
    )

    # Sampling / evaluation
    p.add_argument(
        "--query_sample_frac",
        type=float,
        default=1.0,
        help="Random sample fraction over query rows",
    )
    p.add_argument(
        "--sample_seed", type=int, default=42, help="Seed for query sampling"
    )
    p.add_argument(
        "--query_labels",
        type=str,
        default=None,
        help="Optional true labels (.npy) for accuracy evaluation",
    )

    return p.parse_args()


def load_csr_npz(npz_path: str) -> csr_matrix:
    z = np.load(npz_path, mmap_mode="r")
    for k in ("data", "indices", "indptr", "shape"):
        if k not in z:
            raise ValueError(f"Missing key '{k}' in {npz_path}")
    shape = tuple(int(x) for x in z["shape"])
    return csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)


def pad_dense(x: np.ndarray, d_pad: int) -> np.ndarray:
    if x.shape[1] == d_pad:
        return x
    y = np.zeros((x.shape[0], d_pad), dtype=x.dtype)
    y[:, : x.shape[1]] = x
    return y


def maybe_normalize(x: np.ndarray, metric_type: int):
    # cosine pipeline uses IP metric on L2-normalized vectors
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(x)


def to_runtime_index(index_cpu, gpu: int, use_float16: bool):
    if gpu >= 0:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = use_float16
        co.useFloat16CoarseQuantizer = use_float16
        idx = faiss.index_cpu_to_gpu(res, gpu, index_cpu, co)
        return idx, f"gpu:{gpu}"
    return index_cpu, "cpu"


def add_reference_vectors(
    index, ref_csr, d_index: int, metric_type: int, batch_size: int
):
    n_ref = ref_csr.shape[0]
    log(f"Adding reference vectors: n={n_ref:,}, batch={batch_size:,}")
    for s in tqdm(range(0, n_ref, batch_size), desc="add_ref"):
        e = min(s + batch_size, n_ref)
        xb = ref_csr[s:e].toarray().astype(np.float32, copy=False)
        xb = pad_dense(xb, d_index)
        maybe_normalize(xb, metric_type)
        index.add(xb)
    log(f"Index ntotal after add: {index.ntotal:,}")


def vote_labels(
    nei_idx, nei_dist, ref_labels, n_classes: int, metric_type: int, vote_mode: str
):
    # nei_idx: [B, K], nei_dist: [B, K]
    bsz, k = nei_idx.shape

    valid = (nei_idx >= 0) & (nei_idx < ref_labels.shape[0])

    labels = np.full((bsz, k), -1, dtype=np.int32)
    labels[valid] = ref_labels[nei_idx[valid]]
    valid &= labels >= 0

    if vote_mode == "unweighted":
        weights = valid.astype(np.float32)
    else:
        if metric_type == faiss.METRIC_L2:
            weights = np.zeros_like(nei_dist, dtype=np.float32)
            weights[valid] = 1.0 / (1.0 + nei_dist[valid].astype(np.float32))
        else:
            # IP/cosine: shift each row to non-negative then vote by similarity
            d = nei_dist.astype(np.float32)
            row_min = d.min(axis=1, keepdims=True)
            weights = (d - row_min) + 1e-6
            weights *= valid.astype(np.float32)

    votes = np.zeros((bsz, n_classes), dtype=np.float32)
    row_ids = np.repeat(np.arange(bsz, dtype=np.int64), k)

    flat_valid = valid.ravel()
    np.add.at(
        votes,
        (row_ids[flat_valid], labels.ravel()[flat_valid]),
        weights.ravel()[flat_valid],
    )

    pred = votes.argmax(axis=1).astype(np.int32)
    total = votes.sum(axis=1)
    conf = np.divide(
        votes[np.arange(bsz), pred],
        total,
        out=np.zeros(bsz, dtype=np.float32),
        where=total > 0,
    )

    no_vote = total <= 0
    pred[no_vote] = -1
    conf[no_vote] = 0.0
    return pred, conf


def build_query_indices(
    n_qry: int, max_q: int | None, frac: float, seed: int
) -> np.ndarray:
    n_eff = n_qry if max_q is None else min(n_qry, int(max_q))
    base = np.arange(n_eff, dtype=np.int64)

    if frac <= 0 or frac > 1.0:
        raise ValueError("--query_sample_frac must be in (0, 1]")
    if frac == 1.0:
        return base

    m = int(math.ceil(n_eff * frac))
    m = max(1, min(m, n_eff))
    rng = np.random.default_rng(seed)
    idx = rng.choice(base, size=m, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)


def main():
    args = parse_args()

    if args.output_prefix is None:
        qdir = os.path.dirname(args.query_npz) or "."
        qname = os.path.basename(args.query_npz)
        if qname.endswith(".npz"):
            qname = qname[:-4]
        args.output_prefix = os.path.join(qdir, qname + "_mapped_to_ref")

    out_pred = args.output_prefix + "_clusters.npy"
    out_conf = args.output_prefix + "_confidence.npy"
    out_meta = args.output_prefix + "_summary.json"
    out_qidx = args.output_prefix + "_query_indices.npy"

    log("Loading reference labels...")
    ref_labels = np.load(args.ref_labels, mmap_mode="r").astype(np.int32, copy=False)

    log("Loading reference/query CSR features...")
    ref_csr = load_csr_npz(args.ref_npz)
    qry_csr = load_csr_npz(args.query_npz)

    if args.max_ref_tokens is not None:
        ref_csr = ref_csr[: args.max_ref_tokens]
        ref_labels = ref_labels[: args.max_ref_tokens]
        log(f"Debug cap: ref -> {ref_csr.shape[0]:,}")

    n_ref, d_ref = ref_csr.shape
    n_qry_total, d_qry = qry_csr.shape

    q_idx = build_query_indices(
        n_qry=n_qry_total,
        max_q=args.max_query_tokens,
        frac=float(args.query_sample_frac),
        seed=int(args.sample_seed),
    )
    n_qry_eval = int(q_idx.shape[0])
    if n_qry_eval != n_qry_total:
        log(f"Query evaluation rows: {n_qry_eval:,}/{n_qry_total:,}")
    np.save(out_qidx, q_idx)

    if ref_labels.shape[0] != n_ref:
        raise ValueError(
            f"ref label length {ref_labels.shape[0]} != ref features rows {n_ref}"
        )

    if n_ref == 0 or n_qry_eval == 0:
        raise ValueError("ref or query has zero rows to evaluate")

    if ref_labels.min() < 0:
        log(
            "Warning: ref_labels contains negative values; those will be ignored in voting"
        )

    label_max = int(ref_labels.max())
    n_classes = label_max + 1

    log("Loading trained FAISS index...")
    index_cpu = faiss.read_index(args.trained_index)
    d_index = int(index_cpu.d)
    metric_type = int(index_cpu.metric_type)

    if d_index < d_ref or d_index < d_qry:
        raise ValueError(
            f"Index dim {d_index} is smaller than feature dims ref={d_ref}, query={d_qry}"
        )

    index, device_desc = to_runtime_index(index_cpu, args.gpu, args.use_float16)
    if hasattr(index, "nprobe"):
        index.nprobe = int(args.nprobe)
    log(
        f"Runtime index on {device_desc}; d={d_index}, metric={metric_type}, nprobe={args.nprobe}"
    )

    if int(index.ntotal) != 0:
        log(
            f"Index already has ntotal={index.ntotal:,}; add phase will append more vectors."
        )

    t0 = time.time()
    add_reference_vectors(index, ref_csr, d_index, metric_type, args.add_batch_size)
    t_add = time.time() - t0

    pred = np.empty(n_qry_eval, dtype=np.int32)
    conf = np.empty(n_qry_eval, dtype=np.float32)

    log(
        f"Searching + voting for query vectors: n={n_qry_eval:,}, batch={args.query_batch_size:,}"
    )
    t1 = time.time()
    for s in tqdm(range(0, n_qry_eval, args.query_batch_size), desc="query_map"):
        e = min(s + args.query_batch_size, n_qry_eval)
        rows = q_idx[s:e]

        xq = qry_csr[rows].toarray().astype(np.float32, copy=False)
        xq = pad_dense(xq, d_index)
        maybe_normalize(xq, metric_type)

        dist, idx = index.search(xq, int(args.k))
        p, c = vote_labels(
            idx, dist, ref_labels, n_classes, metric_type, args.vote_mode
        )
        pred[s:e] = p
        conf[s:e] = c
    t_query = time.time() - t1

    log("Saving outputs...")
    np.save(out_pred, pred)
    np.save(out_conf, conf)

    valid = pred >= 0
    pred_counts = (
        np.bincount(pred[valid], minlength=n_classes)
        if valid.any()
        else np.zeros(n_classes, dtype=np.int64)
    )

    summary = {
        "query_npz": args.query_npz,
        "ref_npz": args.ref_npz,
        "ref_labels": args.ref_labels,
        "trained_index": args.trained_index,
        "output_clusters": out_pred,
        "output_confidence": out_conf,
        "output_query_indices": out_qidx,
        "n_ref": int(n_ref),
        "n_query_total": int(n_qry_total),
        "n_query_eval": int(n_qry_eval),
        "d_ref": int(d_ref),
        "d_query": int(d_qry),
        "d_index": int(d_index),
        "k": int(args.k),
        "nprobe": int(args.nprobe),
        "vote_mode": args.vote_mode,
        "query_sample_frac": float(args.query_sample_frac),
        "sample_seed": int(args.sample_seed),
        "mapped_valid": int(valid.sum()),
        "mapped_invalid": int((~valid).sum()),
        "confidence_mean": float(conf[valid].mean()) if valid.any() else 0.0,
        "confidence_median": float(np.median(conf[valid])) if valid.any() else 0.0,
        "top_clusters": [
            {"cluster": int(i), "count": int(pred_counts[i])}
            for i in np.argsort(pred_counts)[::-1][:20]
            if pred_counts[i] > 0
        ],
        "timing_seconds": {
            "add_reference": float(t_add),
            "query_and_vote": float(t_query),
            "total": float(t_add + t_query),
        },
    }

    if args.query_labels:
        qlab_all = np.load(args.query_labels, mmap_mode="r")
        if int(qlab_all.shape[0]) < int(q_idx.max()) + 1:
            raise ValueError(
                "query_labels length is smaller than required by sampled query indices"
            )
        true_lab = qlab_all[q_idx].astype(np.int32, copy=False)

        eval_mask = valid & (true_lab >= 0)
        n_eval = int(eval_mask.sum())
        acc = (
            float((pred[eval_mask] == true_lab[eval_mask]).mean())
            if n_eval > 0
            else 0.0
        )

        summary["accuracy_eval"] = {
            "query_labels": args.query_labels,
            "n_eval": n_eval,
            "accuracy_top1": acc,
        }
        log(f"Accuracy (top1): {acc:.6f} on {n_eval:,} evaluated rows")

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("Done")
    log(f"  clusters:   {out_pred}")
    log(f"  confidence: {out_conf}")
    log(f"  summary:    {out_meta}")
    log(f"  q_idx:      {out_qidx}")


if __name__ == "__main__":
    main()
