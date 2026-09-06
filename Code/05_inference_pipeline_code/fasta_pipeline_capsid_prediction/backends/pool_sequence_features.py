#!/usr/bin/env python3
"""
Convert token-level sparse SAE features to sequence-level mean-pooled sparse features
using token_to_seq mapping arrays.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mean-pool token CSR to sequence CSR by mapping"
    )
    p.add_argument("--token_npz", required=True, help="Token-level CSR npz")
    p.add_argument(
        "--mapping_npz",
        required=True,
        help="NPZ containing token_to_seq and seq_offsets",
    )
    p.add_argument("--out_npz", required=True, help="Output sequence-level CSR npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    token_npz = Path(args.token_npz).expanduser().resolve()
    mapping_npz = Path(args.mapping_npz).expanduser().resolve()
    out_npz = Path(args.out_npz).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading token features: {token_npz}")
    X_token = sp.load_npz(token_npz).tocsr()
    n_tokens, n_feats = X_token.shape
    print(f"      token shape={X_token.shape}, nnz={X_token.nnz}")

    print(f"[2/4] Loading mapping: {mapping_npz}")
    m = np.load(mapping_npz)
    token_to_seq = m["token_to_seq"].astype(np.int64, copy=False)
    seq_offsets = m["seq_offsets"].astype(np.int64, copy=False)

    if token_to_seq.shape[0] != n_tokens:
        raise ValueError(
            f"Length mismatch: token_to_seq={token_to_seq.shape[0]} but token rows={n_tokens}"
        )

    n_seq = seq_offsets.shape[0] - 1
    if n_seq <= 0:
        raise ValueError(f"Invalid seq_offsets length: {seq_offsets.shape[0]}")

    max_seq = int(token_to_seq.max()) + 1
    if max_seq != n_seq:
        print(
            f"[WARN] token_to_seq max+1={max_seq} differs from n_seq={n_seq}, continue anyway."
        )

    lengths = np.diff(seq_offsets).astype(np.float32, copy=False)
    if np.any(lengths <= 0):
        # Robust fallback if offsets contain anomalies
        counts = np.bincount(token_to_seq, minlength=max(max_seq, n_seq)).astype(
            np.float32
        )
        counts[counts <= 0] = 1.0
        inv_len = 1.0 / counts[token_to_seq]
    else:
        inv_len = 1.0 / lengths[token_to_seq]

    print("[3/4] Building pooling operator and multiplying")
    token_idx = np.arange(n_tokens, dtype=np.int64)
    pool_op = sp.csr_matrix(
        (inv_len, (token_to_seq, token_idx)), shape=(n_seq, n_tokens)
    )
    X_seq = pool_op.dot(X_token).tocsr()
    print(f"      seq shape={X_seq.shape}, nnz={X_seq.nnz}")

    print(f"[4/4] Saving: {out_npz}")
    sp.save_npz(out_npz, X_seq)
    print("Done.")


if __name__ == "__main__":
    main()
