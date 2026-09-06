#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(
    "/path/to/vicapsid_data/sae/full_token_backend"
)
PRED = Path(
    "/path/to/vicapsid_data/sae/fold_predictions.csv"
)
RUN_ROOT = Path(
    "/path/to/vicapsid_data/sae/hard_subset_150k"
)
SEED = 42
TARGET_TOTAL = 150000
BTV_RATE = 0.10

# Injected by launcher
if str(RUN_ROOT) == '"""':
    raise RuntimeError("RUN_ROOT placeholder not replaced")

subset_root = RUN_ROOT / "subset_backend"
subset_tokens = subset_root / "token_layers" / "layer_28.tokens.npy"
subset_offsets = subset_root / "seq_token_offsets.npy"
subset_len = subset_root / "seq_token_len.npy"
subset_meta = subset_root / "sample_metadata_filtered.csv"
summary_path = RUN_ROOT / "subset_build_summary.json"

meta = pd.read_csv(BASE / "sample_metadata_filtered.csv")
pred = pd.read_csv(PRED, usecols=["prot_id", "final_label"])
if pred["prot_id"].duplicated().any():
    raise RuntimeError(
        f"Prediction prot_id duplicated: {int(pred['prot_id'].duplicated().sum())}"
    )

merged = meta.merge(pred, on="prot_id", how="left", validate="one_to_one")
if merged["final_label"].isna().any():
    raise RuntimeError(
        f"Missing final_label after merge: {int(merged['final_label'].isna().sum())}"
    )

idx_btv = np.flatnonzero(merged["final_label"].values == "BTV-like")
idx_other = np.flatnonzero(merged["final_label"].values != "BTV-like")

n_btv_keep = int(round(len(idx_btv) * BTV_RATE))
n_btv_keep = max(1, min(n_btv_keep, len(idx_btv)))
n_other_keep = TARGET_TOTAL - n_btv_keep
if n_other_keep > len(idx_other):
    n_other_keep = len(idx_other)
    n_btv_keep = min(len(idx_btv), TARGET_TOTAL - n_other_keep)

rng = np.random.default_rng(SEED)
sel_btv = rng.choice(idx_btv, size=n_btv_keep, replace=False)
sel_other = rng.choice(idx_other, size=n_other_keep, replace=False)
sel = np.concatenate([sel_btv, sel_other])
sel.sort()

if sel.size != TARGET_TOTAL:
    raise RuntimeError(f"Selected {sel.size}, expected {TARGET_TOTAL}")

offsets = np.load(BASE / "seq_token_offsets.npy", mmap_mode="r")
starts = offsets[sel].astype(np.int64)
ends = offsets[sel + 1].astype(np.int64)
lens = (ends - starts).astype(np.int64)
if np.any(lens <= 0):
    bad = int(np.sum(lens <= 0))
    raise RuntimeError(f"Found non-positive token lengths: {bad}")

total_tokens = int(lens.sum())

orig_tokens = np.load(BASE / "token_layers" / "layer_28.tokens.npy", mmap_mode="r")
d_model = int(orig_tokens.shape[1])
out = np.lib.format.open_memmap(
    subset_tokens, mode="w+", dtype=orig_tokens.dtype, shape=(total_tokens, d_model)
)

# Copy sequence token blocks into compact subset token array
write_pos = 0
n = sel.size
for i, (s, e) in enumerate(zip(starts, ends), start=1):
    ln = int(e - s)
    out[write_pos : write_pos + ln] = orig_tokens[s:e]
    write_pos += ln
    if i % 10000 == 0:
        print(f"copied {i}/{n} sequences, tokens={write_pos}", flush=True)

if write_pos != total_tokens:
    raise RuntimeError(f"write_pos mismatch: {write_pos} vs {total_tokens}")

del out

new_offsets = np.empty(n + 1, dtype=np.int64)
new_offsets[0] = 0
np.cumsum(lens, out=new_offsets[1:])
np.save(subset_offsets, new_offsets)
np.save(subset_len, lens.astype(np.int32))

meta_sub = merged.iloc[sel].copy().reset_index(drop=True)
meta_sub.to_csv(subset_meta, index=False)

summary = {
    "target_total": TARGET_TOTAL,
    "selected_total": int(n),
    "selected_btv_like": int((meta_sub["final_label"] == "BTV-like").sum()),
    "selected_other": int((meta_sub["final_label"] != "BTV-like").sum()),
    "selected_btv_ratio": float((meta_sub["final_label"] == "BTV-like").mean()),
    "base_total": int(len(merged)),
    "base_btv_like": int((merged["final_label"] == "BTV-like").sum()),
    "base_other": int((merged["final_label"] != "BTV-like").sum()),
    "tokens_shape": [int(total_tokens), d_model],
    "tokens_dtype": str(orig_tokens.dtype),
    "subset_meta": str(subset_meta),
    "subset_offsets": str(subset_offsets),
    "subset_tokens": str(subset_tokens),
    "seed": SEED,
    "btv_keep_rate": BTV_RATE,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False), flush=True)
