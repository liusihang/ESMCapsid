#!/usr/bin/env python3
"""
Transfer labels from a reference set to a new query set with FAISS kNN.

Priority:
1) GPU FAISS if available and healthy
2) Multi-core CPU FAISS fallback
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAISS kNN label transfer prediction")
    parser.add_argument(
        "--ref_npz", required=True, help="Reference seq features (.npz)"
    )
    parser.add_argument("--ref_labels_csv", required=True, help="Reference labels CSV")
    parser.add_argument(
        "--ref_label_col", default="Cluster", help="Label column in ref_labels_csv"
    )

    parser.add_argument("--query_npz", required=True, help="Query seq features (.npz)")
    parser.add_argument("--query_ids_csv", default=None, help="Optional query IDs CSV")
    parser.add_argument(
        "--query_id_col", default="Seq_ID", help="ID column in query_ids_csv"
    )

    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="knn_transfer", help="Output prefix")

    parser.add_argument("--k", type=int, default=20, help="Neighbors for transfer")
    parser.add_argument(
        "--backend",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Compute backend preference",
    )
    parser.add_argument(
        "--query_batch", type=int, default=5000, help="Batch size for FAISS search"
    )
    parser.add_argument(
        "--cpu_threads", type=int, default=24, help="FAISS CPU thread count"
    )
    parser.add_argument(
        "--include_noise_in_vote",
        action="store_true",
        help="Include label=-1 neighbors in weighted voting (default: exclude)",
    )
    return parser.parse_args()


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    mat /= norms
    return mat


def weighted_vote(
    labels: np.ndarray, sims: np.ndarray, include_noise: bool
) -> tuple[int, float]:
    if not include_noise:
        keep = labels != -1
        labels = labels[keep]
        sims = sims[keep]

    if labels.size == 0:
        return -1, 0.0

    min_sim = float(np.min(sims))
    weights = sims - min_sim + 1e-6

    score = {}
    for lbl, w in zip(labels.tolist(), weights.tolist()):
        key = int(lbl)
        score[key] = score.get(key, 0.0) + float(w)

    pred = max(score, key=score.get)
    total = float(sum(score.values()))
    conf = float(score[pred] / total) if total > 0 else 0.0
    return int(pred), conf


def pick_backend(requested: str) -> str:
    gpu_count = 0
    if hasattr(faiss, "get_num_gpus"):
        try:
            gpu_count = int(faiss.get_num_gpus())
        except Exception:
            gpu_count = 0

    if requested == "gpu":
        if gpu_count < 1:
            raise RuntimeError("backend=gpu requested but no FAISS GPU detected")
        return "gpu"
    if requested == "cpu":
        return "cpu"
    return "gpu" if gpu_count > 0 else "cpu"


def build_index(d: int, backend: str, cpu_threads: int):
    cpu_index = faiss.IndexFlatIP(d)
    if backend == "gpu":
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            return index, {"backend": "gpu", "gpu_id": 0}
        except Exception as e:
            print(f"[WARN] GPU index init failed, fallback to CPU. Reason: {e}")
            faiss.omp_set_num_threads(int(cpu_threads))
            return cpu_index, {
                "backend": "cpu_fallback",
                "cpu_threads": int(cpu_threads),
                "fallback_reason": str(e),
            }

    faiss.omp_set_num_threads(int(cpu_threads))
    return cpu_index, {"backend": "cpu", "cpu_threads": int(cpu_threads)}


def main() -> None:
    args = parse_args()
    t0 = time.time()

    ref_npz = Path(args.ref_npz).expanduser().resolve()
    ref_labels_csv = Path(args.ref_labels_csv).expanduser().resolve()
    query_npz = Path(args.query_npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading reference features: {ref_npz}")
    X_ref = sp.load_npz(ref_npz).tocsr()
    print(f"      Reference shape: {X_ref.shape}")

    print(f"[2/6] Loading reference labels: {ref_labels_csv}")
    df_ref = pd.read_csv(ref_labels_csv, usecols=[args.ref_label_col])
    if len(df_ref) != X_ref.shape[0]:
        raise ValueError(
            f"Row mismatch: ref_labels_csv has {len(df_ref)}, ref_npz has {X_ref.shape[0]}"
        )
    y_ref = df_ref[args.ref_label_col].to_numpy(dtype=np.int64)

    print(f"[3/6] Loading query features: {query_npz}")
    X_query = sp.load_npz(query_npz).tocsr()
    print(f"      Query shape: {X_query.shape}")
    if X_query.shape[1] != X_ref.shape[1]:
        raise ValueError(
            f"Feature mismatch: query dim {X_query.shape[1]} vs reference dim {X_ref.shape[1]}"
        )

    if args.query_ids_csv:
        query_ids_csv = Path(args.query_ids_csv).expanduser().resolve()
        print(f"      Loading query IDs from: {query_ids_csv}")
        df_ids = pd.read_csv(query_ids_csv)
        if args.query_id_col not in df_ids.columns:
            raise ValueError(
                f"query_id_col '{args.query_id_col}' not found in {query_ids_csv}. "
                f"Columns: {list(df_ids.columns)}"
            )
        if len(df_ids) != X_query.shape[0]:
            raise ValueError(
                f"Row mismatch: query_ids_csv has {len(df_ids)}, query_npz has {X_query.shape[0]}"
            )
        query_ids = df_ids[args.query_id_col].astype(str).to_numpy()
    else:
        query_ids = np.arange(X_query.shape[0]).astype(str)

    print("[4/6] Building FAISS index")
    chosen = pick_backend(args.backend)
    index, backend_meta = build_index(X_ref.shape[1], chosen, args.cpu_threads)
    print(f"      Using backend: {backend_meta}")

    X_ref_dense = X_ref.toarray().astype(np.float32, copy=False)
    X_ref_dense = normalize_rows(X_ref_dense)
    index.add(X_ref_dense)
    del X_ref_dense

    print("[5/6] Search + weighted voting")
    pred = np.empty(X_query.shape[0], dtype=np.int64)
    conf = np.zeros(X_query.shape[0], dtype=np.float32)

    for start in range(0, X_query.shape[0], args.query_batch):
        end = min(start + args.query_batch, X_query.shape[0])
        query_dense = X_query[start:end].toarray().astype(np.float32, copy=False)
        query_dense = normalize_rows(query_dense)
        sims, nbr_idx = index.search(query_dense, args.k)
        for i in range(end - start):
            labels_i = y_ref[nbr_idx[i]]
            pred_i, conf_i = weighted_vote(
                labels_i, sims[i], include_noise=args.include_noise_in_vote
            )
            pred[start + i] = pred_i
            conf[start + i] = conf_i
        print(f"      Processed {end}/{X_query.shape[0]}")

    print("[6/6] Saving outputs")
    pred_csv = out_dir / f"{args.prefix}_predictions.csv"
    pd.DataFrame(
        {
            "Seq_ID": query_ids,
            "Pred_Label": pred,
            "Confidence": conf,
        }
    ).to_csv(pred_csv, index=False)

    summary = {
        "reference": {
            "ref_npz": str(ref_npz),
            "ref_labels_csv": str(ref_labels_csv),
            "ref_rows": int(X_ref.shape[0]),
            "ref_dim": int(X_ref.shape[1]),
            "ref_label_col": args.ref_label_col,
            "ref_noise_ratio": float(np.mean(y_ref == -1)),
            "ref_clusters": int(len(set(y_ref) - {-1})),
        },
        "query": {
            "query_npz": str(query_npz),
            "query_rows": int(X_query.shape[0]),
            "query_dim": int(X_query.shape[1]),
            "query_ids_csv": str(args.query_ids_csv) if args.query_ids_csv else None,
            "query_id_col": args.query_id_col,
        },
        "knn": {
            "k": int(args.k),
            "backend_request": args.backend,
            "backend_used": backend_meta,
            "query_batch": int(args.query_batch),
            "cpu_threads": int(args.cpu_threads),
            "include_noise_in_vote": bool(args.include_noise_in_vote),
        },
        "prediction_stats": {
            "pred_noise_ratio": float(np.mean(pred == -1)),
            "pred_clusters": int(len(set(pred) - {-1})),
            "avg_confidence": float(np.mean(conf)),
        },
        "outputs": {"predictions_csv": str(pred_csv)},
        "runtime_seconds": float(time.time() - t0),
    }

    summary_path = out_dir / f"{args.prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved predictions: {pred_csv}")
    print(f"Saved summary: {summary_path}")
    print(f"Pred noise ratio: {summary['prediction_stats']['pred_noise_ratio']:.4f}")
    print(f"Pred clusters: {summary['prediction_stats']['pred_clusters']}")
    print(f"Avg confidence: {summary['prediction_stats']['avg_confidence']:.4f}")


if __name__ == "__main__":
    main()
