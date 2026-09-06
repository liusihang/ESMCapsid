#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess and sample residue representations for SAE training.

Outputs include per-epoch sampled arrays, optional initialization samples,
preprocessing statistics, a machine-readable configuration and data-quality metrics.
"""

import os
import sys
import json
import time
import gc
import argparse
import threading
import shutil
import tempfile
import atexit
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================

# ==========================================
def set_numpy_threads(n):
    for env_var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[env_var] = str(n)


# ==========================================

# ==========================================
DEFAULT_CONFIG = {
    
    "input_dir": "/path/to/hpc_scratch/Temp/MLMfinetunedAllCapsidTokenNPY_LMean",
    "layer_idx": 0,
    "meta_path": "/path/to/hpc_scratch/Temp/MLMfinetunedAllCapsidTokenNPY_LMean/sample_metadata_filtered.csv",
    
    "output_dir": "/path/to/hpc_scratch/Temp/MLMfinetunedAllCapsidTokenNPY_LMean/PreprocessedData",
    
    "use_local_staging": False,
    "local_staging_dir": "/tmp",
    
    "samples_per_epoch": None,  
    "steps_per_epoch": 1024,  
    "batch_size": 4096,
    "epochs": 12,
    
    "target_keywords": ["IMGVR", "MGY"],
    "balance_ratio": 3.5,
    
    "mu_sample_size": 200000,
    
    "generate_init_data": True,  
    "init_sample_size": 100000,  # Initialization samples
    
    "l2_normalize": True,
    "save_dtype": "float32",
    "random_seed": 42,
    
    "parallel_epochs": 6,
    "num_threads": 12,
    "chunk_size": 100000,
    "sort_before_read": True,
    
    "validate_data": True,  
    "max_validation_samples": 10000,  
    
    "resume": True,  
}

# ==========================================

# ==========================================
_print_lock = threading.Lock()


def log_info(msg, prefix="INFO"):
    t = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{t}] [{prefix}] {msg}", flush=True)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def l2_normalize_np(x, eps=1e-9):
    """NumPy L2 normalization"""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def human_readable_size(size_bytes):
    """Convert a byte count to a readable size"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def estimate_storage(
    samples_per_epoch, epochs, d_in, dtype="float32", include_init=False, init_size=0
):
    """Estimate storage requirements"""
    bytes_per_sample = d_in * (2 if dtype == "float16" else 4)
    total_bytes = samples_per_epoch * epochs * bytes_per_sample
    if include_init:
        total_bytes += init_size * bytes_per_sample
    return total_bytes


def validate_data_quality(data, name="data"):
    """Validate data quality"""
    issues = []

    
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        issues.append(f"Contains {nan_count:,} NaN values")

    
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        issues.append(f"Contains {inf_count:,} Inf values")

    
    zero_rows = np.all(data == 0, axis=1).sum()
    if zero_rows > 0:
        issues.append(f"Contains {zero_rows:,} all-zero rows")

    
    stats = {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "zero_rows": int(zero_rows),
    }

    if issues:
        log_info(f"  Data quality issues in {name}:", prefix="WARN")
        for issue in issues:
            log_info(f"   - {issue}", prefix="WARN")
    else:
        log_info(f"✓ {name} quality check passed", prefix="VALID")

    return stats, len(issues) == 0


# ==========================================

# ==========================================
class DataStager:
    """Local staging manager"""

    def __init__(self, config):
        self.use_staging = config.get("use_local_staging", False)
        self.local_dir = config.get("local_staging_dir", "/tmp")
        self.input_dir = config["input_dir"]
        self.layer_idx = config["layer_idx"]
        self.meta_path = config["meta_path"]

        self.staging_dir = None
        self.staged_paths = {}

    def setup(self):
        """Stage data on the local compute node"""
        original_paths = {
            "tokens": os.path.join(
                self.input_dir, "token_layers", f"layer_{self.layer_idx}.tokens.npy"
            ),
            "offsets": os.path.join(self.input_dir, "seq_token_offsets.npy"),
            "meta": self.meta_path,
        }

        if not self.use_staging:
            log_info("Using network storage directly (no staging)")
            return original_paths

        self.staging_dir = tempfile.mkdtemp(prefix="sae_staging_", dir=self.local_dir)
        log_info(f"Staging to local node: {self.staging_dir}")
        atexit.register(self.cleanup)

        start_time = time.time()

        for key, src in original_paths.items():
            if not os.path.exists(src):
                log_info(f"Warning: {src} not found, skipping", prefix="WARN")
                self.staged_paths[key] = src
                continue

            file_size = os.path.getsize(src)
            dst = os.path.join(self.staging_dir, os.path.basename(src))

            log_info(f"Copying {key}: {human_readable_size(file_size)}")

            copy_start = time.time()
            shutil.copyfile(src, dst)
            copy_time = time.time() - copy_start

            throughput = file_size / (1024**2) / copy_time if copy_time > 0 else 0
            log_info(f"  → {dst} ({copy_time:.1f}s, {throughput:.1f} MB/s)")

            self.staged_paths[key] = dst

        total_time = time.time() - start_time
        log_info(f"Staging complete in {total_time:.1f}s")

        return self.staged_paths

    def cleanup(self):
        """Clean the staging directory"""
        if self.staging_dir and os.path.exists(self.staging_dir):
            log_info(f"Cleaning up staging dir: {self.staging_dir}")
            try:
                shutil.rmtree(self.staging_dir)
            except Exception as e:
                log_info(f"Cleanup warning: {e}", prefix="WARN")


# ==========================================

# ==========================================
class BalancedTokenSampler:
    def __init__(self, meta_df, offsets, target_keywords, ratio=2.0, seed=42):
        self.offsets = offsets.astype(np.int64)
        self.base_seed = seed

        seq_lens = (self.offsets[1:] - self.offsets[:-1]).astype(np.int64)
        valid_seq_mask = seq_lens > 0

        ids = meta_df.iloc[:, 0].astype(str)
        pattern = "|".join(target_keywords)
        mask_target = ids.str.contains(pattern, case=False, regex=True).values

        indices = np.arange(len(meta_df))
        self.group_a = indices[mask_target & valid_seq_mask]
        self.group_b = indices[(~mask_target) & valid_seq_mask]
        self.valid_indices = indices[valid_seq_mask]

        n_a, n_b = len(self.group_a), len(self.group_b)
        n_valid = len(self.valid_indices)

        log_info(f"Sampler: Target={n_a:,}, Others={n_b:,}, Total={n_valid:,}")
        log_info(
            f"Expected ratio: Target {ratio / (ratio + 1) * 100:.1f}% : Others {1 / (ratio + 1) * 100:.1f}%"
        )

        self.weights = np.zeros(len(meta_df), dtype=np.float64)

        if n_valid == 0:
            raise RuntimeError("No valid sequences found.")

        if n_a == 0 or n_b == 0:
            self.weights[self.valid_indices] = 1.0 / n_valid
        else:
            w_a = (ratio / (ratio + 1.0)) / n_a
            w_b = (1.0 / (ratio + 1.0)) / n_b
            self.weights[self.group_a] = w_a
            self.weights[self.group_b] = w_b

        self.weights[~valid_seq_mask] = 0.0

        weight_sum = self.weights.sum()
        if weight_sum > 0:
            self.weights = self.weights / weight_sum
        else:
            self.weights[self.valid_indices] = 1.0 / n_valid

    def get_statistics(self):
        """Return sampler statistics"""
        return {
            "n_target": int(len(self.group_a)),
            "n_others": int(len(self.group_b)),
            "n_valid": int(len(self.valid_indices)),
            "target_ratio": float(len(self.group_a) / max(len(self.valid_indices), 1)),
        }


# ==========================================

# ==========================================
def read_mmap_sorted_chunks(
    X_mmap, indices, chunk_size, normalize=True, dtype=np.float32
):
    """
    Sort-then-Read with chunking
    Returns a NumPy array.
    """
    n_samples = len(indices)
    d_in = X_mmap.shape[1]

    
    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]

    
    output = np.empty((n_samples, d_in), dtype=dtype)

    
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min((ci + 1) * chunk_size, n_samples)

        chunk_sorted_idx = sorted_indices[start:end]
        chunk_output_pos = sort_order[start:end]

        
        chunk_data = X_mmap[chunk_sorted_idx].astype(np.float32)

        
        if normalize:
            chunk_data = l2_normalize_np(chunk_data)

        
        output[chunk_output_pos] = chunk_data.astype(dtype)

    return output


# ==========================================

# ==========================================
def process_single_epoch(
    epoch_idx,
    tokens_path,
    output_dir,
    sampler_weights,
    sampler_offsets,
    samples_per_epoch,
    d_in,
    config,
    resume=False,
):
    """Process one epoch"""
    epoch_filename = f"sampled_tokens_ep{epoch_idx:02d}.npy"
    epoch_path = os.path.join(output_dir, epoch_filename)

    
    if resume and os.path.exists(epoch_path):
        log_info(
            f"Epoch {epoch_idx + 1} already exists, skipping",
            prefix=f"EP{epoch_idx + 1:02d}",
        )
        return epoch_path

    ep_t0 = time.time()
    log_info(f"Epoch {epoch_idx + 1} started", prefix=f"EP{epoch_idx + 1:02d}")

    save_dtype = np.float16 if config["save_dtype"] == "float16" else np.float32
    chunk_size = config.get("chunk_size", 100000)

    
    epoch_seed = config["random_seed"] + epoch_idx * 1000
    rng = np.random.default_rng(epoch_seed)

    seq_indices = rng.choice(
        len(sampler_weights), size=samples_per_epoch, replace=True, p=sampler_weights
    )
    starts = sampler_offsets[seq_indices]
    lens = (sampler_offsets[seq_indices + 1] - starts).astype(np.int64)
    offsets_within = (rng.random(samples_per_epoch) * lens).astype(np.int64)
    token_indices = starts + offsets_within

    
    X_mmap = np.load(tokens_path, mmap_mode="r")

    # Sort-then-Read
    log_info(
        f"Reading {samples_per_epoch:,} samples...", prefix=f"EP{epoch_idx + 1:02d}"
    )
    t_read = time.time()

    output_data = read_mmap_sorted_chunks(
        X_mmap,
        token_indices,
        chunk_size,
        normalize=config["l2_normalize"],
        dtype=save_dtype,
    )

    read_time = time.time() - t_read
    throughput_gb = (
        (samples_per_epoch * d_in * 4) / (1024**3) / read_time if read_time > 0 else 0
    )
    log_info(
        f"Read complete: {read_time:.1f}s ({throughput_gb:.2f} GB/s)",
        prefix=f"EP{epoch_idx + 1:02d}",
    )

    
    np.save(epoch_path, output_data)

    file_size_gb = os.path.getsize(epoch_path) / (1024**3)
    elapsed = time.time() - ep_t0
    log_info(
        f"Saved: {epoch_filename} ({file_size_gb:.2f} GB, {elapsed:.1f}s)",
        prefix=f"EP{epoch_idx + 1:02d}",
    )

    del output_data, X_mmap
    gc.collect()

    return epoch_path


# ==========================================

# ==========================================
def generate_init_dataset(
    tokens_path, sampler_weights, sampler_offsets, n_samples, d_in, config, output_dir
):
    """
    Generate a dataset for K-Means++ initialization
    Use an independent random seed for initialization sampling
    """
    log_info(f"Generating initialization dataset ({n_samples:,} samples)...")

    init_seed = config["random_seed"] + 999999  
    rng = np.random.default_rng(init_seed)

    
    seq_indices = rng.choice(
        len(sampler_weights), size=n_samples, replace=True, p=sampler_weights
    )
    starts = sampler_offsets[seq_indices]
    lens = (sampler_offsets[seq_indices + 1] - starts).astype(np.int64)
    offsets_within = (rng.random(n_samples) * lens).astype(np.int64)
    token_indices = starts + offsets_within

    
    X_mmap = np.load(tokens_path, mmap_mode="r")

    init_data = read_mmap_sorted_chunks(
        X_mmap,
        token_indices,
        chunk_size=config.get("chunk_size", 100000),
        normalize=config["l2_normalize"],
        dtype=np.float32,  
    )

    
    init_path = os.path.join(output_dir, "init_samples.npy")
    np.save(init_path, init_data)

    file_size = os.path.getsize(init_path) / (1024**2)
    log_info(f"Init dataset saved: {init_path} ({file_size:.1f} MB)")

    del init_data
    gc.collect()

    return init_path


# ==========================================

# ==========================================
def run_preprocessing(config):
    log_info("=" * 70)
    log_info("SAE Pre-processing & Sampling v2 (Optimized for Training)")
    log_info("=" * 70)

    
    num_threads = config.get("num_threads", 4)
    parallel_epochs = config.get("parallel_epochs", 1)
    set_numpy_threads(max(1, num_threads // parallel_epochs))

    
    stager = DataStager(config)
    staged_paths = stager.setup()

    tokens_path = staged_paths["tokens"]
    offsets_path = staged_paths["offsets"]
    meta_path = staged_paths["meta"]

    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    log_info(f"")
    log_info(f"Input tokens: {tokens_path}")
    log_info(f"Output dir: {output_dir}")
    log_info(f"")

    
    log_info("Loading metadata and offsets...")
    meta_df = pd.read_csv(meta_path)
    offsets = np.load(offsets_path)

    X_mmap = np.load(tokens_path, mmap_mode="r")
    n_tokens, d_in = X_mmap.shape
    log_info(
        f"Token data: shape={X_mmap.shape}, size={human_readable_size(os.path.getsize(tokens_path))}"
    )

    
    if config.get("samples_per_epoch"):
        samples_per_epoch = config["samples_per_epoch"]
        steps_per_epoch = samples_per_epoch // config.get("batch_size", 4096)
    else:
        steps_per_epoch = config["steps_per_epoch"]
        batch_size = config["batch_size"]
        samples_per_epoch = steps_per_epoch * batch_size

    epochs = config["epochs"]

    
    include_init = config.get("generate_init_data", False)
    init_size = config.get("init_sample_size", 0) if include_init else 0

    estimated_bytes = estimate_storage(
        samples_per_epoch, epochs, d_in, config["save_dtype"], include_init, init_size
    )

    log_info(f"")
    log_info(f"{'=' * 50}")
    log_info(f"Data configuration")
    log_info(f"{'=' * 50}")
    log_info(f"Samples per epoch:  {samples_per_epoch:,}")
    log_info(f"Batch size:       {config.get('batch_size', 'auto')}")
    log_info(f"Steps per epoch:  {steps_per_epoch:,}")
    log_info(f"Epochs:           {epochs}")
    log_info(f"Input dimension (d_in):      {d_in}")
    log_info(f"Data type:         {config['save_dtype']}")
    if include_init:
        log_info(f"Initialization samples:     {init_size:,}")
    log_info(f"Estimated total storage:       {human_readable_size(estimated_bytes)}")
    log_info(f"{'=' * 50}")
    log_info(f"")

    
    if estimated_bytes > 500 * 1024**3:
        log_info("  Estimated storage exceeds 500 GB.", prefix="WARN")
        log_info("   Reduce samples_per_epoch or use float16.", prefix="WARN")
        log_info("")

    
    log_info(f"Computing global mean from {config['mu_sample_size']:,} samples...")
    mu_t0 = time.time()

    rng = np.random.default_rng(config["random_seed"])
    mu_indices = rng.choice(
        n_tokens, min(config["mu_sample_size"], n_tokens), replace=False
    )
    mu_indices_sorted = np.sort(mu_indices)

    mu_samples = X_mmap[mu_indices_sorted].astype(np.float32)

    if config["l2_normalize"]:
        mu_samples = l2_normalize_np(mu_samples)

    global_mean = mu_samples.mean(axis=0)

    
    if config.get("validate_data", True):
        mu_stats, mu_valid = validate_data_quality(mu_samples, "global_mean_samples")

    del mu_samples

    mu_path = os.path.join(output_dir, "global_mean.npy")
    np.save(mu_path, global_mean.astype(np.float32))

    mu_time = time.time() - mu_t0
    log_info(f"Global mean computed in {mu_time:.1f}s, saved to: {mu_path}")

    
    log_info("Creating balanced sampler...")
    sampler = BalancedTokenSampler(
        meta_df=meta_df,
        offsets=offsets,
        target_keywords=config["target_keywords"],
        ratio=config["balance_ratio"],
        seed=config["random_seed"],
    )

    sampler_stats = sampler.get_statistics()

    
    init_path = None
    if config.get("generate_init_data", False):
        init_size = config.get("init_sample_size", 100000)
        init_path = generate_init_dataset(
            tokens_path,
            sampler.weights,
            sampler.offsets,
            init_size,
            d_in,
            config,
            output_dir,
        )

    log_info(f"")
    log_info(
        f"Sort-then-Read: {'ON ✓' if config.get('sort_before_read', True) else 'OFF'}"
    )
    log_info(f"Parallel epochs: {parallel_epochs}, Threads: {num_threads}")
    log_info(f"Resume mode: {'ON' if config.get('resume', False) else 'OFF'}")
    log_info(f"")

    
    total_t0 = time.time()
    epoch_files = []
    resume = config.get("resume", False)

    if parallel_epochs > 1:
        log_info(f"Processing {epochs} epochs (max {parallel_epochs} parallel)...")

        with ProcessPoolExecutor(max_workers=parallel_epochs) as executor:
            futures = {}
            for ep in range(epochs):
                future = executor.submit(
                    process_single_epoch,
                    ep,
                    tokens_path,
                    output_dir,
                    sampler.weights,
                    sampler.offsets,
                    samples_per_epoch,
                    d_in,
                    config,
                    resume,
                )
                futures[future] = ep

            for future in as_completed(futures):
                ep = futures[future]
                try:
                    epoch_path = future.result()
                    epoch_files.append((ep, epoch_path))
                except Exception as e:
                    log_info(f"Epoch {ep + 1} failed: {e}", prefix="ERROR")
                    import traceback

                    traceback.print_exc()

        epoch_files.sort(key=lambda x: x[0])
        epoch_files = [path for _, path in epoch_files]
    else:
        for ep in range(epochs):
            try:
                epoch_path = process_single_epoch(
                    ep,
                    tokens_path,
                    output_dir,
                    sampler.weights,
                    sampler.offsets,
                    samples_per_epoch,
                    d_in,
                    config,
                    resume,
                )
                epoch_files.append(epoch_path)
            except Exception as e:
                log_info(f"Epoch {ep + 1} failed: {e}", prefix="ERROR")
                import traceback

                traceback.print_exc()

    
    validation_stats = {}
    if config.get("validate_data", True) and epoch_files:
        log_info("Validating first epoch data...")
        val_data = np.load(epoch_files[0])
        n_val = min(config.get("max_validation_samples", 10000), len(val_data))
        val_sample = val_data[:n_val]

        val_stats, val_valid = validate_data_quality(val_sample, "epoch_0_sample")
        validation_stats = val_stats

        del val_data, val_sample
        gc.collect()

    
    preprocess_config = {
        
        "input_tokens_path": tokens_path,
        "input_meta_path": meta_path,
        "output_dir": output_dir,
        
        "d_in": int(d_in),
        "n_tokens_total": int(n_tokens),
        
        "n_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": config.get("batch_size", samples_per_epoch // steps_per_epoch),
        "samples_per_epoch": samples_per_epoch,
        
        "l2_normalized": config["l2_normalize"],
        "save_dtype": config["save_dtype"],
        
        "target_keywords": config["target_keywords"],
        "balance_ratio": config["balance_ratio"],
        "random_seed": config["random_seed"],
        
        "epoch_files": epoch_files,
        "global_mean_path": mu_path,
        "init_samples_path": init_path,
        
        "sampler_stats": sampler_stats,
        
        "validation_stats": validation_stats,
        
        "created_at": datetime.now().isoformat(),
        "sort_before_read": config.get("sort_before_read", True),
        "preprocessing_version": "v2",
    }

    config_path = os.path.join(output_dir, "preprocess_config.json")
    with open(config_path, "w") as f:
        json.dump(preprocess_config, f, indent=2)

    
    stats_summary = {
        "data_shape": {
            "n_tokens": int(n_tokens),
            "d_in": int(d_in),
            "n_sequences": int(len(meta_df)),
        },
        "sampling": sampler_stats,
        "storage": {
            "total_samples": samples_per_epoch * epochs,
            "total_bytes": int(
                sum(os.path.getsize(f) for f in epoch_files if os.path.exists(f))
            ),
            "dtype": config["save_dtype"],
        },
        "validation": validation_stats,
    }

    stats_path = os.path.join(output_dir, "data_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)

    
    total_time = time.time() - total_t0
    total_size = sum(os.path.getsize(f) for f in epoch_files if os.path.exists(f))

    log_info("=" * 70)
    log_info("Pre-processing Complete!")
    log_info("=" * 70)
    log_info(f"Total time:        {total_time:.1f}s ({total_time / 60:.1f} min)")
    log_info(f"Total size:        {human_readable_size(total_size)}")
    log_info(f"Throughput:        {total_size / (1024**2) / total_time:.1f} MB/s")
    log_info(f"Epochs processed:  {len(epoch_files)}/{epochs}")
    log_info(f"")
    log_info(f"Output files:")
    log_info(f"  Config:          {config_path}")
    log_info(f"  Statistics:      {stats_path}")
    log_info(f"  Global mean:     {mu_path}")
    if init_path:
        log_info(f"  Init samples:    {init_path}")
    log_info(f"  Epoch data:      {len(epoch_files)} files")
    log_info("=" * 70)

    return output_dir


# ==========================================

# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE Pre-processing v2 (Optimized for Training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--meta_path", type=str)
    parser.add_argument("--layer_idx", type=int, default=0)

    
    parser.add_argument(
        "--use_staging", action="store_true", help="Stage input data on the local node"
    )
    parser.add_argument("--staging_dir", type=str, default="/tmp")

    
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples_per_epoch", type=int, help="Samples per epoch")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)

    
    parser.add_argument(
        "--gen_init_data", action="store_true", help="Generate K-Means++ initialization data"
    )
    parser.add_argument(
        "--init_samples", type=int, default=100000, help="Initialization sample count"
    )
    parser.add_argument("--no_validate", action="store_true", help="Skip data validation")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume behavior")

    
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float16", "float32"]
    )
    parser.add_argument("--no_normalize", action="store_true")

    
    parser.add_argument("--parallel_epochs", type=int, default=6)
    parser.add_argument("--num_threads", type=int, default=12)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--no_sort", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    
    if args.input_dir:
        config["input_dir"] = args.input_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.meta_path:
        config["meta_path"] = args.meta_path
    if args.layer_idx:
        config["layer_idx"] = args.layer_idx

    if args.use_staging:
        config["use_local_staging"] = True
    if args.staging_dir:
        config["local_staging_dir"] = args.staging_dir

    if args.epochs:
        config["epochs"] = args.epochs
    if args.samples_per_epoch:
        config["samples_per_epoch"] = args.samples_per_epoch
    if args.steps:
        config["steps_per_epoch"] = args.steps
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seed:
        config["random_seed"] = args.seed
    if args.dtype:
        config["save_dtype"] = args.dtype
    if args.no_normalize:
        config["l2_normalize"] = False

    
    if args.gen_init_data:
        config["generate_init_data"] = True
    if args.init_samples:
        config["init_sample_size"] = args.init_samples
    if args.no_validate:
        config["validate_data"] = False
    if args.no_resume:
        config["resume"] = False

    config["parallel_epochs"] = args.parallel_epochs
    config["num_threads"] = args.num_threads
    config["chunk_size"] = args.chunk_size
    config["sort_before_read"] = not args.no_sort

    run_preprocessing(config)


if __name__ == "__main__":
    main()
