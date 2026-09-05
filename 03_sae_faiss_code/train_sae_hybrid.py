#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train, evaluate and apply the configurable top-k sparse autoencoder."""

import os
import sys
import json
import time
import gc
import argparse
import shutil
import tempfile
import atexit
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# ==========================================

# ==========================================
DEFAULT_CONFIG = {
    "data_dir": "",
    "labels_csv_path": "",
    "output_root": "./SAE_Results_Hybrid",
    "sweep_dims": [2302, 4604],
    "sweep_ks": [32, 64],
    "lr": 1e-4,
    "batch_size": 7196,
    "early_stop_patience": 5,
    "early_stop_delta": 1e-5,
    
    "do_mean_center": True,
    "do_rescale": True,
    "use_robust_scaling": True,
    "clip_extreme_values": True,
    "clip_std_multiplier": 5.0,
    "use_local_staging": True,
    "local_staging_dir": "/tmp",
    "num_workers": 4,
    "use_amp": True,
    "use_torch_compile": True,
    "infer_chunk_size": 4096,
    
    "loss_type": "huber",
    "huber_delta": 5,
    "sparsity_coeff": 0.05,
    "sparsity_target": 0.01,  
    "decoder_norm_coeff": 0.0005,
    "decoder_norm_interval": 200,
    
    "aux_loss_coeff": 1 / 1024,
    "aux_k_multiplier": 2,
    "min_aux_k": 48,
    
    "dead_neuron_threshold": 500,  
    "rare_rate_per_step": 0.01,
    "resample_interval_epochs": 1,
    "resample_limit_ratio": 0.1,
    
    "warmup_ratio": 0.1,
    "min_lr_ratio": 0.1,
    "grad_clip_norm": 1.0,
    "init_method": "kmeans_pp",
    "enc_init_scale": 0.1,  
    
    "detailed_neuron_stats": True,
    "neuron_stats_interval": 5,
    "diagnose_per_dim": True,
    "diagnose_interval": 10,
    "log_grad_norm_interval": 500,  
    
    "stats_sample_ratio": 0.2,
    "stats_max_rows_per_file": 50000,
    "stats_seed": 42,
}


# ==========================================

# ==========================================
def log_info(msg, prefix="INFO"):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] [{prefix}] {msg}", flush=True)


def ensure_dir(p):
    if not p or str(p).strip() == "":
        raise ValueError(f"Invalid directory path: '{p}'")
    os.makedirs(p, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================

# ==========================================
def compute_reconstruction_loss(pred, target, loss_type="huber", delta=10.0):
    """Compute the configured reconstruction loss"""
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    elif loss_type == "huber":
        return F.huber_loss(pred, target, delta=delta)
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=delta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ==========================================
# DataStager
# ==========================================
class DataStager:
    """Local staging manager"""

    def __init__(self, config, preprocess_config):
        self.use_staging = config.get("use_local_staging", False)
        self.local_dir = config.get("local_staging_dir", "/dev/shm")
        self.preprocess_config = preprocess_config
        self.staging_dir = None
        self.original_paths = {}

    def setup(self):
        orig_tokens_path = self.preprocess_config.get("input_tokens_path", "")
        self.original_paths = {
            "tokens": orig_tokens_path,
            "meta": self.preprocess_config.get("input_meta_path", ""),
        }

        if not self.use_staging:
            log_info("Using original paths directly (no staging)")
            return self.preprocess_config

        if not orig_tokens_path or (not os.path.exists(orig_tokens_path)):
            log_info(
                f"Original tokens path not found: {orig_tokens_path}", prefix="ERROR"
            )
            log_info("Disabling staging and continuing...", prefix="WARN")
            self.use_staging = False
            return self.preprocess_config

        self.staging_dir = tempfile.mkdtemp(prefix="sae_staging_", dir=self.local_dir)
        log_info(f"Staging to: {self.staging_dir}")
        atexit.register(self.cleanup)

        staged_tokens = os.path.join(
            self.staging_dir, os.path.basename(orig_tokens_path)
        )
        log_info(
            f"Copying tokens ({os.path.getsize(orig_tokens_path) / (1024**3):.2f} GB)..."
        )
        shutil.copyfile(orig_tokens_path, staged_tokens)

        base_dir = os.path.dirname(orig_tokens_path)
        offsets_candidates = [
            os.path.join(base_dir, "seq_token_offsets.npy"),
            os.path.join(os.path.dirname(base_dir), "seq_token_offsets.npy"),
        ]
        staged_offsets = None
        for off_path in offsets_candidates:
            if os.path.exists(off_path):
                staged_offsets = os.path.join(self.staging_dir, "seq_token_offsets.npy")
                shutil.copyfile(off_path, staged_offsets)
                log_info(f"Copied offsets from: {off_path}")
                break

        updated = dict(self.preprocess_config)
        updated["input_tokens_path"] = staged_tokens
        if staged_offsets:
            updated["staged_offsets_path"] = staged_offsets
        return updated

    def get_original_tokens_path(self):
        return self.original_paths.get("tokens", "")

    def cleanup(self):
        if self.staging_dir and os.path.exists(self.staging_dir):
            try:
                shutil.rmtree(self.staging_dir)
                log_info(f"Cleaned up staging: {self.staging_dir}")
            except Exception as e:
                log_info(f"Cleanup warning: {e}", prefix="WARN")


# ==========================================
# Dataset
# ==========================================
class PreprocessedEpochDataset(Dataset):
    def __init__(self, npy_path, batch_size, steps):
        try:
            self.data = np.load(npy_path, mmap_mode="r")
        except Exception:
            self.data = np.load(npy_path)

        self.batch_size = int(batch_size)
        self.steps = int(steps)

        expected = self.batch_size * self.steps
        if self.data.shape[0] < expected:
            self.steps = max(1, self.data.shape[0] // self.batch_size)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        st = idx * self.batch_size
        ed = st + self.batch_size
        sample = self.data[st:ed]
        if isinstance(self.data, np.memmap):
            return torch.from_numpy(np.ascontiguousarray(sample)).float()
        return torch.from_numpy(sample).float()


# ==========================================

# ==========================================
def estimate_stats_from_chunks(preprocess_config, config):
    """Support standard and robust IQR scaling"""
    sample_ratio = float(config.get("stats_sample_ratio", 0.2))
    max_rows_per_file = int(config.get("stats_max_rows_per_file", 50000))
    seed = int(config.get("stats_seed", 42))
    use_robust = bool(config.get("use_robust_scaling", False))
    clip_extreme = bool(config.get("clip_extreme_values", True))
    clip_std_mult = float(config.get("clip_std_multiplier", 5.0))
    eps = 1e-6

    epoch_files = preprocess_config["epoch_files"]
    n_files = len(epoch_files)
    n_files_to_read = max(1, int(n_files * sample_ratio))

    rng = np.random.default_rng(seed)
    files_to_scan = rng.choice(
        epoch_files, size=n_files_to_read, replace=False
    ).tolist()
    log_info(
        f"Estimating stats from {n_files_to_read}/{n_files} chunks (robust={use_robust})..."
    )

    all_data = []
    total_rows = 0

    for fpath in files_to_scan:
        X = np.load(fpath, mmap_mode="r")
        n = X.shape[0]
        take = min(n, max_rows_per_file)
        if take < n:
            idx = rng.choice(n, size=take, replace=False)
            B = np.asarray(X[idx], dtype=np.float32)
        else:
            B = np.asarray(X[:take], dtype=np.float32)
        all_data.append(B)
        total_rows += B.shape[0]

    all_data = np.concatenate(all_data, axis=0)
    d_in = all_data.shape[1]

    mu = all_data.mean(axis=0).astype(np.float32)
    std = all_data.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps)

    median = np.median(all_data, axis=0).astype(np.float32)
    q75 = np.percentile(all_data, 75, axis=0).astype(np.float32)
    q25 = np.percentile(all_data, 25, axis=0).astype(np.float32)
    iqr = np.maximum(q75 - q25, eps).astype(np.float32)

    data_min = all_data.min(axis=0).astype(np.float32)
    data_max = all_data.max(axis=0).astype(np.float32)

    if use_robust:
        center = median
        scale = iqr
        alpha = (1.0 / scale).astype(np.float32)
        log_info(f"Using ROBUST scaling: center=median, scale=IQR")
    else:
        center = mu
        scale = std
        alpha = (1.0 / scale).astype(np.float32)
        log_info(f"Using STANDARD scaling: center=mean, scale=std")

    if clip_extreme:
        clip_min = (mu - clip_std_mult * std).astype(np.float32)
        clip_max = (mu + clip_std_mult * std).astype(np.float32)
        log_info(f"Clip bounds: mean ± {clip_std_mult} * std")
    else:
        clip_min = np.full(d_in, -np.inf, dtype=np.float32)
        clip_max = np.full(d_in, np.inf, dtype=np.float32)

    log_info(f"Stats computed from {total_rows:,} rows, {d_in} dims")
    log_info(f"  Mean: min={mu.min():.3f}, max={mu.max():.3f}")
    log_info(
        f"  Std:  min={std.min():.3f}, med={np.median(std):.3f}, max={std.max():.3f}"
    )
    log_info(f"  Data range: [{data_min.min():.1f}, {data_max.max():.1f}]")

    extreme_dims = np.where((data_min < mu - 10 * std) | (data_max > mu + 10 * std))[0]
    if len(extreme_dims) > 0:
        log_info(
            f"   Detected {len(extreme_dims)} extreme dims (>10σ): {extreme_dims[:10]}...",
            prefix="WARN",
        )

    return {
        "mu": mu,
        "std": std,
        "median": median,
        "iqr": iqr,
        "center": center,
        "scale": scale,
        "alpha": alpha,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "data_min": data_min,
        "data_max": data_max,
        "n_total_rows_used": int(total_rows),
        "use_robust": use_robust,
    }


# ==========================================

# ==========================================
def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================

# ==========================================
def compute_detailed_neuron_stats(sae, threshold_steps=1000):
    """
    v2.4: Track primary and auxiliary activation statistics separately
    """
    
    if hasattr(sae, "main_activation_count"):
        activation_counts = sae.main_activation_count.cpu().numpy()
        steps_since_fired = sae.main_steps_since_fired.cpu().numpy()
    else:
        
        activation_counts = sae.neuron_activation_count.cpu().numpy()
        steps_since_fired = sae.steps_since_fired.cpu().numpy()

    n_total = len(activation_counts)
    n_active = (activation_counts > 0).sum()
    n_dead = (steps_since_fired > threshold_steps).sum()

    if n_active > 0:
        active_counts = activation_counts[activation_counts > 0]
        activation_stats = {
            "mean": float(active_counts.mean()),
            "std": float(active_counts.std()),
            "min": float(active_counts.min()),
            "max": float(active_counts.max()),
            "median": float(np.median(active_counts)),
            "q25": float(np.percentile(active_counts, 25)),
            "q75": float(np.percentile(active_counts, 75)),
        }
    else:
        activation_stats = {}

    recent_inactive = (steps_since_fired > threshold_steps // 2) & (
        steps_since_fired <= threshold_steps
    )
    critically_inactive = steps_since_fired > threshold_steps * 2

    sorted_counts = np.sort(activation_counts)
    n = len(sorted_counts)
    if np.sum(sorted_counts) > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (
            n + 1
        ) / n
    else:
        gini = 0.0

    
    total_steps = sae.total_steps.item() if hasattr(sae, "total_steps") else 1
    activation_rates = activation_counts / (total_steps + 1e-8)

    
    rates_normalized = activation_rates / (activation_rates.sum() + 1e-10)
    activation_entropy = -np.sum(rates_normalized * np.log(rates_normalized + 1e-10))

    stats = {
        "n_total": int(n_total),
        "n_active": int(n_active),
        "n_dead": int(n_dead),
        "n_at_risk": int(recent_inactive.sum()),
        "n_critically_dead": int(critically_inactive.sum()),
        "active_ratio": float(n_active / n_total),
        "dead_ratio": float(n_dead / n_total),
        "at_risk_ratio": float(recent_inactive.sum() / n_total),
        "activation_stats": activation_stats,
        "gini_coefficient": float(gini),
        "activation_entropy": float(activation_entropy),
        "total_activations": int(activation_counts.sum()),
        "max_steps_since_fired": int(steps_since_fired.max())
        if steps_since_fired.size
        else 0,
        "threshold_used": int(threshold_steps),
    }

    
    stats["dead_by_threshold"] = {}
    for t in [100, 500, 1000, 2000, 5000]:
        dead_at_t = (steps_since_fired > t).sum()
        stats["dead_by_threshold"][t] = int(dead_at_t)

    return stats


def log_detailed_neuron_stats(stats, epoch, prefix="NEURO"):
    log_info("=" * 60, prefix=prefix)
    log_info(f"Detailed Neuron Statistics @ Epoch {epoch}", prefix=prefix)
    log_info("=" * 60, prefix=prefix)

    log_info(f"Total neurons:        {stats['n_total']:,}", prefix=prefix)
    log_info(
        f"Active neurons:       {stats['n_active']:,} ({stats['active_ratio'] * 100:.2f}%)",
        prefix=prefix,
    )
    log_info(
        f"Dead neurons:         {stats['n_dead']:,} ({stats['dead_ratio'] * 100:.2f}%) [threshold={stats['threshold_used']}]",
        prefix=prefix,
    )
    log_info(
        f"At-risk neurons:      {stats['n_at_risk']:,} ({stats['at_risk_ratio'] * 100:.2f}%)",
        prefix=prefix,
    )
    log_info(f"Critically dead:      {stats['n_critically_dead']:,}", prefix=prefix)
    log_info(f"Total activations:    {stats['total_activations']:,}", prefix=prefix)
    log_info(
        f"Max inactive steps:   {stats.get('max_steps_since_fired', 0)}", prefix=prefix
    )

    
    log_info("", prefix=prefix)
    log_info("Dead neurons by threshold:", prefix=prefix)
    for t, count in stats.get("dead_by_threshold", {}).items():
        pct = count / stats["n_total"] * 100
        log_info(f"  threshold={t}: {count} ({pct:.2f}%)", prefix=prefix)

    if stats["activation_stats"]:
        log_info("", prefix=prefix)
        log_info("Activation distribution (active neurons only):", prefix=prefix)
        a = stats["activation_stats"]
        log_info(f"  Mean:   {a['mean']:.1f}", prefix=prefix)
        log_info(f"  Median: {a['median']:.1f}", prefix=prefix)
        log_info(f"  Std:    {a['std']:.1f}", prefix=prefix)
        log_info(f"  Range:  [{a['min']:.0f}, {a['max']:.0f}]", prefix=prefix)
        log_info(f"  Q25-Q75: [{a['q25']:.0f}, {a['q75']:.0f}]", prefix=prefix)

    log_info("", prefix=prefix)
    log_info(
        f"Gini coefficient:     {stats['gini_coefficient']:.4f} (0=equal, 1=unequal)",
        prefix=prefix,
    )
    log_info(f"Activation entropy:   {stats['activation_entropy']:.4f}", prefix=prefix)

    if stats["dead_ratio"] < 0.01:
        health = "✓ EXCELLENT"
    elif stats["dead_ratio"] < 0.05:
        health = "✓ GOOD"
    elif stats["dead_ratio"] < 0.15:
        health = " FAIR"
    else:
        health = "✗ POOR"

    log_info(f"Overall health:       {health}", prefix=prefix)
    log_info("=" * 60, prefix=prefix)


# ==========================================

# ==========================================
def compute_per_dim_reconstruction_stats(sae, data_batch, config):
    """
    v2.4: Use the current clipping attributes
    """
    sae.eval()
    device = next(sae.parameters()).device

    with torch.no_grad():
        x = data_batch.to(device)
        x_cs = sae._center_scale(x)

        
        

        topk_idx, topk_vals = sae.encode(x)
        recon = sae.decode(topk_idx, topk_vals)

        per_dim_mse = ((x_cs - recon) ** 2).mean(dim=0)
        per_dim_var = x_cs.var(dim=0)
        per_dim_ev = 1.0 - per_dim_mse / (per_dim_var + 1e-8)

        total_mse = per_dim_mse.mean()
        total_var = per_dim_var.mean()
        total_ev = 1.0 - total_mse / (total_var + 1e-8)

        sorted_ev, sorted_idx = torch.sort(per_dim_ev)

        return {
            "per_dim_mse": per_dim_mse.cpu().numpy(),
            "per_dim_var": per_dim_var.cpu().numpy(),
            "per_dim_ev": per_dim_ev.cpu().numpy(),
            "worst_dims": sorted_idx[:20].cpu().numpy(),
            "worst_ev": sorted_ev[:20].cpu().numpy(),
            "best_dims": sorted_idx[-20:].flip(0).cpu().numpy(),
            "best_ev": sorted_ev[-20:].flip(0).cpu().numpy(),
            "total_ev": float(total_ev.item()),
            "total_mse": float(total_mse.item()),
            "median_dim_ev": float(per_dim_ev.median().item()),
            "dims_below_50pct_ev": int((per_dim_ev < 0.5).sum().item()),
            "dims_below_0_ev": int((per_dim_ev < 0).sum().item()),
        }


def log_per_dim_stats(stats, epoch, prefix="DIMS"):
    log_info("=" * 60, prefix=prefix)
    log_info(f"Per-Dimension Reconstruction @ Epoch {epoch}", prefix=prefix)
    log_info("=" * 60, prefix=prefix)

    log_info(f"Total Explained Variance: {stats['total_ev'] * 100:.2f}%", prefix=prefix)
    log_info(f"Median Dim EV: {stats['median_dim_ev'] * 100:.2f}%", prefix=prefix)
    log_info(f"Dims with EV < 50%: {stats['dims_below_50pct_ev']}", prefix=prefix)
    log_info(
        f"Dims with EV < 0%: {stats['dims_below_0_ev']} (reconstruction worse than mean)",
        prefix=prefix,
    )

    log_info("", prefix=prefix)
    log_info("Worst reconstructed dimensions:", prefix=prefix)
    for i, (dim, ev) in enumerate(
        zip(stats["worst_dims"][:10], stats["worst_ev"][:10])
    ):
        log_info(f"  #{i + 1}: dim={dim}, EV={ev * 100:.1f}%", prefix=prefix)

    log_info("", prefix=prefix)
    log_info("Best reconstructed dimensions:", prefix=prefix)
    for i, (dim, ev) in enumerate(zip(stats["best_dims"][:5], stats["best_ev"][:5])):
        log_info(f"  #{i + 1}: dim={dim}, EV={ev * 100:.1f}%", prefix=prefix)

    log_info("=" * 60, prefix=prefix)


# ==========================================

# ==========================================
def log_gradient_norms(model, prefix="GRAD"):
    """Record parameter gradient norms"""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    log_info("Gradient norms:", prefix=prefix)
    for name, norm in grad_norms.items():
        status = "" if norm > 10 or norm < 1e-7 else "✓"
        log_info(f"  {status} {name}: {norm:.6f}", prefix=prefix)

    return grad_norms


# ==========================================

# ==========================================
class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder with v2.4 fixes
    """

    def __init__(self, d_in, d_dict, topk, aux_k=None, enc_init_scale=0.1):
        super().__init__()
        self.d_in = int(d_in)
        self.d_dict = int(d_dict)
        self.topk = int(topk)
        self.aux_k = int(aux_k) if aux_k is not None else max(self.topk // 2, 32)
        self.enc_init_scale = enc_init_scale  

        self.W_enc = nn.Parameter(torch.empty(self.d_in, self.d_dict))
        self.b_enc = nn.Parameter(torch.zeros(self.d_dict))

        self.W_dec = nn.Parameter(torch.empty(self.d_dict, self.d_in))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))

        
        self.register_buffer("b_pre", torch.zeros(self.d_in))
        self.register_buffer("alpha", torch.ones(self.d_in))

        
        self.register_buffer("clip_min", torch.full((self.d_in,), -float("inf")))
        self.register_buffer("clip_max", torch.full((self.d_in,), float("inf")))
        self.do_clip = False

        
        
        self.register_buffer("main_activation_count", torch.zeros(self.d_dict))
        self.register_buffer("main_steps_since_fired", torch.zeros(self.d_dict))

        
        self.register_buffer("aux_activation_count", torch.zeros(self.d_dict))

        
        self.register_buffer("neuron_activation_count", torch.zeros(self.d_dict))
        self.register_buffer("steps_since_fired", torch.zeros(self.d_dict))

        self.register_buffer("total_steps", torch.tensor(0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=np.sqrt(5))
        with torch.no_grad():
            self.W_dec[:] = F.normalize(self.W_dec, dim=1)

    def set_clip_bounds(self, clip_min, clip_max):
        """Set clipping bounds in normalized space"""
        with torch.no_grad():
            self.clip_min.copy_(torch.from_numpy(clip_min))
            self.clip_max.copy_(torch.from_numpy(clip_max))
            self.do_clip = True

    @torch.no_grad()
    def init_with_data(self, data_batch, method="kmeans_pp"):
        data_processed = self._center_scale(data_batch)
        n_samples = data_processed.shape[0]

        if n_samples < self.d_dict:
            log_info(
                f"Warning: samples ({n_samples}) < d_dict ({self.d_dict}), using cycling",
                prefix="WARN",
            )
            indices = torch.arange(self.d_dict, device=data_batch.device) % n_samples
            samples = data_processed[indices]
        elif method == "random":
            indices = torch.randperm(n_samples, device=data_batch.device)[: self.d_dict]
            samples = data_processed[indices]
        elif method == "kmeans_pp":
            log_info("Using K-Means++ style initialization...")
            samples = self._kmeans_pp_init(data_processed, self.d_dict)
        elif method == "high_norm":
            norms = data_processed.norm(dim=1)
            _, indices = torch.topk(norms, min(self.d_dict, n_samples))
            samples = data_processed[indices]
            if len(indices) < self.d_dict:
                extra = self.d_dict - len(indices)
                extra_idx = torch.randint(n_samples, (extra,), device=data_batch.device)
                samples = torch.cat([samples, data_processed[extra_idx]], dim=0)
        else:
            raise ValueError(f"Unknown init method: {method}")

        samples = F.normalize(samples, dim=1)

        self.W_dec.data.copy_(samples)
        self.W_enc.data.copy_(samples.T * self.enc_init_scale)  
        self.b_enc.data.zero_()
        self.b_dec.data.zero_()

        self._reset_neuron_stats()

        log_info(f"Initialized SAE with {self.d_dict} vectors using '{method}' method")

    def _reset_neuron_stats(self):
        """Reset feature-usage statistics"""
        self.main_activation_count.zero_()
        self.main_steps_since_fired.zero_()
        self.aux_activation_count.zero_()
        self.neuron_activation_count.zero_()
        self.steps_since_fired.zero_()
        self.total_steps.zero_()

    def _kmeans_pp_init(self, data, n_centers, max_samples=50000):
        device = data.device
        n = data.shape[0]

        if n > max_samples:
            idx = torch.randperm(n, device=device)[:max_samples]
            data = data[idx]
            n = max_samples

        centers = []
        first_idx = torch.randint(n, (1,), device=device)
        centers.append(data[first_idx])

        for i in range(1, n_centers):
            if i % 2000 == 0:
                log_info(f"  K-Means++ init: {i}/{n_centers}")

            center_stack = torch.cat(centers, dim=0)

            chunk_size = 10000
            min_dists = []
            for j in range(0, n, chunk_size):
                chunk = data[j : j + chunk_size]
                dists = torch.cdist(chunk, center_stack)
                min_dists.append(dists.min(dim=1).values)

            min_dists = torch.cat(min_dists)

            probs = min_dists**2
            probs = probs / (probs.sum() + 1e-10)

            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.ones(n, device=device) / n

            idx = torch.multinomial(probs, 1)
            centers.append(data[idx])

        return torch.cat(centers, dim=0)

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec[:] = F.normalize(self.W_dec, dim=1)

    def get_decoder_norm_loss(self):
        """Decoder norm penalty"""
        dec_norms = self.W_dec.norm(dim=1)
        return ((dec_norms - 1.0) ** 2).mean()

    def _center_scale(self, x):
        """Center and scale inputs"""
        x_cs = (x - self.b_pre) * self.alpha
        if self.do_clip:
            x_cs = torch.clamp(x_cs, self.clip_min, self.clip_max)
        return x_cs

    def encode(self, x):
        x_cs = self._center_scale(x)
        pre_acts = x_cs @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        topk_vals, topk_idx = torch.topk(acts, self.topk, dim=1)
        return topk_idx, topk_vals

    def decode(self, topk_idx, topk_vals):
        W_subset = self.W_dec[topk_idx]
        recon = (topk_vals.unsqueeze(-1) * W_subset).sum(dim=1) + self.b_dec
        return recon

    def decode_to_original_space(self, topk_idx, topk_vals):
        """v2.4: Decode into the original input space"""
        recon_normalized = self.decode(topk_idx, topk_vals)
        return recon_normalized / self.alpha + self.b_pre

    def forward_train(self, x):
        """Training forward pass"""
        x_cs = self._center_scale(x)
        pre_acts = x_cs @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)

        big_k = min(self.topk + self.aux_k, self.d_dict)
        big_vals, big_idx = torch.topk(acts, big_k, dim=1)

        topk_vals = big_vals[:, : self.topk]
        topk_idx = big_idx[:, : self.topk]

        aux_topk_vals = big_vals[:, self.topk : self.topk + self.aux_k]
        aux_topk_idx = big_idx[:, self.topk : self.topk + self.aux_k]

        recon = self.decode(topk_idx, topk_vals)

        residual = (x_cs - recon).detach()
        aux_recon = self.decode(aux_topk_idx, aux_topk_vals)

        return {
            "target": x_cs,
            "recon": recon,
            "residual": residual,
            "aux_recon": aux_recon,
            "topk_idx": topk_idx,
            "aux_topk_idx": aux_topk_idx,
            "topk_vals": topk_vals,
            "pre_acts": pre_acts,
            "acts": acts,  
        }

    @torch.no_grad()
    def update_dead_neuron_stats(self, main_idx, aux_idx=None):
        """
        v2.4: Update primary and auxiliary activation statistics separately

        Args:
            main_idx: indices of primary top-k activations
            aux_idx: optional auxiliary activation indices
        """
        self.total_steps += 1

        
        main_activation = torch.zeros(self.d_dict, device=main_idx.device)
        flat_main_idx = main_idx.flatten()
        main_activation.scatter_add_(
            0, flat_main_idx, torch.ones_like(flat_main_idx, dtype=torch.float)
        )

        self.main_activation_count += main_activation
        self.main_steps_since_fired += 1
        main_fired_mask = main_activation > 0
        self.main_steps_since_fired[main_fired_mask] = 0

        
        if aux_idx is not None:
            aux_activation = torch.zeros(self.d_dict, device=aux_idx.device)
            flat_aux_idx = aux_idx.flatten()
            aux_activation.scatter_add_(
                0, flat_aux_idx, torch.ones_like(flat_aux_idx, dtype=torch.float)
            )
            self.aux_activation_count += aux_activation

        
        combined_activation = main_activation.clone()
        if aux_idx is not None:
            combined_activation += aux_activation

        self.neuron_activation_count += combined_activation
        self.steps_since_fired += 1
        combined_fired_mask = combined_activation > 0
        self.steps_since_fired[combined_fired_mask] = 0

    @torch.no_grad()
    def get_dead_neuron_mask(self, threshold_steps=1000, use_main_only=True):
        """
        v2.4: Choose primary or combined activations for inactive-feature detection
        """
        if use_main_only:
            return self.main_steps_since_fired > threshold_steps
        else:
            return self.steps_since_fired > threshold_steps

    @torch.no_grad()
    def get_dead_neuron_ratio(self, threshold_steps=1000, use_main_only=True):
        dead_mask = self.get_dead_neuron_mask(threshold_steps, use_main_only)
        return dead_mask.float().mean().item()

    @torch.no_grad()
    def get_usage_ratios(
        self, threshold_steps=1000, rare_rate_per_step=0.01, use_main_only=True
    ):
        """
        v2.4: Return detailed feature-usage statistics
        """
        if use_main_only:
            steps_since = self.main_steps_since_fired
            activation_count = self.main_activation_count
        else:
            steps_since = self.steps_since_fired
            activation_count = self.neuron_activation_count

        dead_ratio = (steps_since > threshold_steps).float().mean().item()
        never_ratio = (activation_count == 0).float().mean().item()
        rate = activation_count / (self.total_steps.float() + 1e-8)
        rare_ratio = (rate < rare_rate_per_step).float().mean().item()
        max_inactive = float(steps_since.max().item()) if steps_since.numel() else 0.0

        return dead_ratio, never_ratio, rare_ratio, max_inactive

    @torch.no_grad()
    def get_activation_distribution(self):
        """v2.4: Return activation-distribution statistics"""
        total_steps = self.total_steps.item() + 1e-8

        main_rates = (self.main_activation_count / total_steps).cpu().numpy()
        aux_rates = (self.aux_activation_count / total_steps).cpu().numpy()
        combined_rates = (self.neuron_activation_count / total_steps).cpu().numpy()

        return {
            "main_rates": main_rates,
            "aux_rates": aux_rates,
            "combined_rates": combined_rates,
            "main_only_neurons": int(
                (self.main_activation_count > 0).sum().item()
                - ((self.main_activation_count > 0) & (self.aux_activation_count > 0))
                .sum()
                .item()
            ),
            "aux_only_neurons": int(
                (self.aux_activation_count > 0).sum().item()
                - ((self.main_activation_count > 0) & (self.aux_activation_count > 0))
                .sum()
                .item()
            ),
            "both_neurons": int(
                ((self.main_activation_count > 0) & (self.aux_activation_count > 0))
                .sum()
                .item()
            ),
            "never_activated": int((self.neuron_activation_count == 0).sum().item()),
        }

    @torch.no_grad()
    def resample_dead_neurons(
        self, data_batch, threshold_steps=1000, resample_limit=None, use_main_only=True
    ):
        """v2.4: Resample using primary-activation statistics"""
        dead_mask = self.get_dead_neuron_mask(threshold_steps, use_main_only)
        dead_indices = dead_mask.nonzero().squeeze(-1)
        n_dead = len(dead_indices)

        if n_dead == 0:
            return 0

        if resample_limit is not None:
            n_to_resample = min(n_dead, resample_limit)
            dead_indices = dead_indices[:n_to_resample]
        else:
            n_to_resample = n_dead

        x_cs = self._center_scale(data_batch)
        topk_idx, topk_vals = self.encode(data_batch)
        recon = self.decode(topk_idx, topk_vals)

        recon_error = ((x_cs - recon) ** 2).sum(dim=1)

        n_samples_needed = min(n_to_resample, len(data_batch))
        _, high_loss_idx = torch.topk(recon_error, n_samples_needed)

        replacement_vectors = x_cs[high_loss_idx]
        replacement_vectors = F.normalize(replacement_vectors, dim=1)

        n_actual = min(len(replacement_vectors), len(dead_indices))
        dead_indices = dead_indices[:n_actual]
        replacement_vectors = replacement_vectors[:n_actual]

        self.W_dec.data[dead_indices] = replacement_vectors
        self.W_enc.data[:, dead_indices] = replacement_vectors.T * self.enc_init_scale
        self.b_enc.data[dead_indices] = 0

        
        self.main_steps_since_fired[dead_indices] = 0
        self.main_activation_count[dead_indices] = 0
        self.steps_since_fired[dead_indices] = 0
        self.neuron_activation_count[dead_indices] = 0

        return n_actual

    def get_training_metrics(self, forward_output):
        topk_vals = forward_output["topk_vals"]
        topk_idx = forward_output["topk_idx"]
        target = forward_output["target"]
        recon = forward_output["recon"]

        with torch.no_grad():
            l0 = (topk_vals > 0).float().sum(dim=1).mean()

            mse = F.mse_loss(recon, target)
            var = target.var(unbiased=False)
            explained_var = 1.0 - mse / (var + 1e-8)

            unique_neurons = topk_idx.unique().numel()

            return {
                "l0": float(l0.item()),
                "explained_var": float(explained_var.item()),
                "batch_active_neurons": int(unique_neurons),
                "batch_active_ratio": float(unique_neurons / self.d_dict),
                "mse": float(mse.item()),
                "var": float(var.item()),
            }


# ==========================================

# ==========================================
def train_one_run(run_dir, d_dict, k, preprocess_config, stats, config):
    log_info(f"{'=' * 60}")
    log_info(f"Training SAE: D={d_dict}, K={k}")
    log_info(f"{'=' * 60}")

    model_path = os.path.join(run_dir, "sae_model.pt")
    if os.path.exists(model_path):
        log_info(
            f"Model already exists, skipping training: {model_path}", prefix="SKIP"
        )
        return model_path

    device = get_device()
    use_amp = config["use_amp"]

    d_in = preprocess_config["d_in"]

    aux_k_mult = float(config.get("aux_k_multiplier", 1.5))
    min_aux_k = int(config.get("min_aux_k", 64))
    aux_k = max(int(k * aux_k_mult), min_aux_k)

    enc_init_scale = float(config.get("enc_init_scale", 0.1))
    sae = TopKSAE(d_in, d_dict, k, aux_k=aux_k, enc_init_scale=enc_init_scale).to(
        device
    )

    
    with torch.no_grad():
        if config["do_mean_center"]:
            center = stats["center"] if "center" in stats else stats["mu"]
            sae.b_pre.copy_(torch.from_numpy(center).to(device))
        if config["do_rescale"]:
            sae.alpha.copy_(torch.from_numpy(stats["alpha"]).to(device))

        if config.get("clip_extreme_values", False):
            raw_clip_min = stats["clip_min"]
            raw_clip_max = stats["clip_max"]
            center = stats["center"] if "center" in stats else stats["mu"]
            alpha = stats["alpha"]

            norm_clip_min = (raw_clip_min - center) * alpha
            norm_clip_max = (raw_clip_max - center) * alpha

            sae.set_clip_bounds(
                norm_clip_min.astype(np.float32), norm_clip_max.astype(np.float32)
            )
            log_info(f"Clip bounds set in normalized space")

    
    log_info("Performing Data-Driven Initialization...")
    init_dataset = PreprocessedEpochDataset(
        preprocess_config["epoch_files"][0],
        config["batch_size"],
        preprocess_config["steps_per_epoch"],
    )

    init_samples = []
    needed = min(d_dict * 2, 100000)
    loader_iter = iter(DataLoader(init_dataset, batch_size=None, num_workers=0))

    collected = 0
    try:
        while collected < needed:
            batch = next(loader_iter)
            init_samples.append(batch)
            collected += batch.shape[0]
            if len(init_samples) > 50:
                break
    except StopIteration:
        pass

    if init_samples:
        full_batch = torch.cat(init_samples, dim=0).to(device)
        init_method = config.get("init_method", "kmeans_pp")
        sae.init_with_data(full_batch, method=init_method)
        del full_batch, init_samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log_info("Failed to load data for init, using random.", prefix="WARN")

    
    sae_compiled = sae
    if config["use_torch_compile"]:
        try:
            sae_compiled = torch.compile(sae, mode="reduce-overhead")
            log_info("Model compiled with torch.compile")
        except Exception as e:
            log_info(f"torch.compile failed: {e}", prefix="WARN")
            sae_compiled = sae

    opt = torch.optim.AdamW(sae.parameters(), lr=config["lr"], betas=(0.9, 0.999))

    epoch_files = preprocess_config["epoch_files"]
    n_epochs = preprocess_config["n_epochs"]
    steps_per_epoch = preprocess_config["steps_per_epoch"]
    batch_size = preprocess_config["batch_size"]

    total_steps = n_epochs * steps_per_epoch
    warmup_ratio = config.get("warmup_ratio", 0.1)
    warmup_steps = min(int(total_steps * warmup_ratio), 2000)
    min_lr_ratio = config.get("min_lr_ratio", 0.1)

    scheduler = get_lr_scheduler(opt, warmup_steps, total_steps, min_lr_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    
    loss_type = config.get("loss_type", "huber")
    huber_delta = float(config.get("huber_delta", 10.0))
    aux_loss_coeff = float(config.get("aux_loss_coeff", 1 / 128))
    sparsity_coeff = float(config.get("sparsity_coeff", 0.0))
    sparsity_target = float(config.get("sparsity_target", 0.01))
    decoder_norm_coeff = float(config.get("decoder_norm_coeff", 0.01))
    decoder_norm_interval = int(config.get("decoder_norm_interval", 200))

    
    dead_neuron_threshold_cfg = int(config.get("dead_neuron_threshold", 1000))
    dead_neuron_threshold_auto = max(3 * steps_per_epoch, 500)
    dead_neuron_threshold = min(dead_neuron_threshold_cfg, dead_neuron_threshold_auto)
    if dead_neuron_threshold != dead_neuron_threshold_cfg:
        log_info(
            f"Dead threshold adjusted: cfg={dead_neuron_threshold_cfg} -> auto={dead_neuron_threshold}",
            prefix="WARN",
        )

    rare_rate_per_step = float(config.get("rare_rate_per_step", 0.01))

    resample_interval = steps_per_epoch * int(config.get("resample_interval_epochs", 1))
    resample_limit = int(d_dict * float(config.get("resample_limit_ratio", 0.1)))
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

    
    detailed_stats = bool(config.get("detailed_neuron_stats", True))
    stats_interval = int(config.get("neuron_stats_interval", 5))
    diagnose_per_dim = bool(config.get("diagnose_per_dim", True))
    diagnose_interval = int(config.get("diagnose_interval", 10))
    log_grad_interval = int(config.get("log_grad_norm_interval", 500))

    best_loss = float("inf")
    patience = 0
    global_step = 0

    history = {
        "main_loss": [],
        "aux_loss": [],
        "explained_var": [],
        "dead_ratio": [],
        "dead_ratio_main_only": [],  
        "never_ratio": [],
        "rare_ratio": [],
        "max_inactive": [],
        "lr": [],
        "decoder_norm_loss": [],
        "sparsity_loss": [],
        "detailed_neuron_stats": [],
        "per_dim_stats": [],
        "activation_distribution": [],  
    }

    log_info(f"Training config:")
    log_info(
        f"  Epochs: {n_epochs}, Steps/epoch: {steps_per_epoch}, Batch: {batch_size}"
    )
    log_info(f"  Warmup: {warmup_steps} steps, Total: {total_steps} steps")
    log_info(f"  Loss: {loss_type} (delta={huber_delta})")
    log_info(f"  AuxK: {aux_k}, Aux coeff: {aux_loss_coeff}")
    log_info(f"  Sparsity coeff: {sparsity_coeff}, target: {sparsity_target}")
    log_info(
        f"  Decoder norm coeff: {decoder_norm_coeff}, interval: {decoder_norm_interval}"
    )
    log_info(f"  Dead threshold: {dead_neuron_threshold} steps")
    log_info(f"  Clip extreme: {config.get('clip_extreme_values', False)}")
    log_info(f"  Enc init scale: {enc_init_scale}")

    
    diag_data_cache = None

    for ep in range(n_epochs):
        ep_main_loss = 0.0
        ep_aux_loss = 0.0
        ep_decoder_norm_loss = 0.0
        ep_sparsity_loss = 0.0

        
        ep_total_mse = 0.0
        ep_total_var = 0.0

        t0 = time.time()

        current_file = epoch_files[ep % len(epoch_files)]
        dataset = PreprocessedEpochDataset(current_file, batch_size, steps_per_epoch)
        loader = DataLoader(
            dataset, batch_size=None, num_workers=config["num_workers"], pin_memory=True
        )

        sae_compiled.train()

        for i, x in enumerate(loader):
            x = x.to(device, non_blocking=True)

            if diag_data_cache is None and i == 0:
                diag_data_cache = x.clone()

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = sae_compiled.forward_train(x)

                main_loss = compute_reconstruction_loss(
                    out["recon"], out["target"], loss_type=loss_type, delta=huber_delta
                )
                aux_loss = compute_reconstruction_loss(
                    out["aux_recon"],
                    out["residual"],
                    loss_type=loss_type,
                    delta=huber_delta,
                )

                loss = main_loss + aux_loss_coeff * aux_loss

                
                if sparsity_coeff > 0:
                    
                    # sparsity_loss = F.relu(out['pre_acts']).mean()

                    
                    activation_rate = (out["acts"] > 0).float().mean()
                    sparsity_loss = (activation_rate - sparsity_target) ** 2

                    loss = loss + sparsity_coeff * sparsity_loss
                    ep_sparsity_loss += float(sparsity_loss.item())

                if decoder_norm_coeff > 0:
                    dec_norm_loss = sae.get_decoder_norm_loss()
                    loss = loss + decoder_norm_coeff * dec_norm_loss
                    ep_decoder_norm_loss += float(dec_norm_loss.item())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=grad_clip_norm)

            scaler.step(opt)
            scaler.update()

            scheduler.step()
            global_step += 1

            # v2.4: Update primary and auxiliary activation statistics separately
            sae.update_dead_neuron_stats(
                main_idx=out["topk_idx"], aux_idx=out["aux_topk_idx"]
            )

            if (i + 1) % decoder_norm_interval == 0:
                sae.normalize_decoder()

            
            if global_step % resample_interval == 0 and global_step > warmup_steps:
                n_resampled = sae.resample_dead_neurons(
                    x,
                    threshold_steps=dead_neuron_threshold,
                    resample_limit=resample_limit,
                    use_main_only=True,  
                )
                if n_resampled > 0:
                    log_info(
                        f"  Step {global_step}: Resampled {n_resampled} dead neurons (main-only)"
                    )

            ep_main_loss += float(main_loss.item())
            ep_aux_loss += float(aux_loss.item())

            
            with torch.no_grad():
                batch_mse = F.mse_loss(out["recon"], out["target"], reduction="sum")
                batch_var = out["target"].var(unbiased=False) * out["target"].numel()
                ep_total_mse += batch_mse.item()
                ep_total_var += batch_var.item()

            
            if global_step % log_grad_interval == 0:
                log_gradient_norms(sae, prefix="GRAD")

        
        avg_main_loss = ep_main_loss / steps_per_epoch
        avg_aux_loss = ep_aux_loss / steps_per_epoch
        avg_explained_var = 1.0 - ep_total_mse / (ep_total_var + 1e-8)
        avg_decoder_norm_loss = (
            ep_decoder_norm_loss / steps_per_epoch if decoder_norm_coeff > 0 else 0
        )
        avg_sparsity_loss = (
            ep_sparsity_loss / steps_per_epoch if sparsity_coeff > 0 else 0
        )

        
        dead_ratio_main, never_ratio, rare_ratio, max_inactive = sae.get_usage_ratios(
            threshold_steps=dead_neuron_threshold,
            rare_rate_per_step=rare_rate_per_step,
            use_main_only=True,
        )
        dead_ratio_combined, _, _, _ = sae.get_usage_ratios(
            threshold_steps=dead_neuron_threshold,
            rare_rate_per_step=rare_rate_per_step,
            use_main_only=False,
        )

        current_lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        
        if detailed_stats and (ep + 1) % stats_interval == 0:
            neuron_stats = compute_detailed_neuron_stats(sae, dead_neuron_threshold)
            log_detailed_neuron_stats(neuron_stats, ep + 1)

            
            stats_to_save = {}
            for k, v in neuron_stats.items():
                if isinstance(v, np.ndarray):
                    stats_to_save[k] = v.tolist()
                elif isinstance(v, dict):
                    stats_to_save[k] = {
                        kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()
                    }
                else:
                    stats_to_save[k] = v

            history["detailed_neuron_stats"].append(
                {"epoch": ep + 1, "stats": stats_to_save}
            )

            
            act_dist = sae.get_activation_distribution()
            
            act_dist_serializable = {}
            for k, v in act_dist.items():
                if isinstance(v, np.ndarray):
                    
                    act_dist_serializable[k] = {
                        "mean": float(v.mean()),
                        "std": float(v.std()),
                        "min": float(v.min()),
                        "max": float(v.max()),
                    }
                else:
                    act_dist_serializable[k] = (
                        int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v
                    )

            log_info(
                f"Activation distribution: main_only={act_dist['main_only_neurons']}, "
                f"aux_only={act_dist['aux_only_neurons']}, "
                f"both={act_dist['both_neurons']}, "
                f"never={act_dist['never_activated']}",
                prefix="ACTIV",
            )
            history["activation_distribution"].append(
                {"epoch": ep + 1, "dist": act_dist_serializable}
            )

        
        if (
            diagnose_per_dim
            and (ep + 1) % diagnose_interval == 0
            and diag_data_cache is not None
        ):
            dim_stats = compute_per_dim_reconstruction_stats(
                sae, diag_data_cache, config
            )
            log_per_dim_stats(dim_stats, ep + 1)
            history["per_dim_stats"].append(
                {
                    "epoch": ep + 1,
                    "stats": {
                        "total_ev": dim_stats["total_ev"],
                        "median_dim_ev": dim_stats["median_dim_ev"],
                        "dims_below_50pct_ev": dim_stats["dims_below_50pct_ev"],
                        "dims_below_0_ev": dim_stats["dims_below_0_ev"],
                        "worst_dims": dim_stats["worst_dims"].tolist(),
                        "worst_ev": dim_stats["worst_ev"].tolist(),
                    },
                }
            )

        
        log_info(
            f"Epoch {ep + 1}/{n_epochs} | "
            f"Main: {avg_main_loss:.6f} | "
            f"Aux: {avg_aux_loss:.6f} | "
            f"EV: {avg_explained_var:.3f} | "
            f"Dead(main): {dead_ratio_main * 100:.1f}% | "
            f"Dead(all): {dead_ratio_combined * 100:.1f}% | "
            f"MaxInact: {max_inactive:.0f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {dt:.1f}s"
        )

        history["main_loss"].append(float(avg_main_loss))
        history["aux_loss"].append(float(avg_aux_loss))
        history["explained_var"].append(float(avg_explained_var))
        history["dead_ratio"].append(float(dead_ratio_combined))
        history["dead_ratio_main_only"].append(float(dead_ratio_main))
        history["never_ratio"].append(float(never_ratio))
        history["rare_ratio"].append(float(rare_ratio))
        history["max_inactive"].append(float(max_inactive))
        history["lr"].append(float(current_lr))
        history["decoder_norm_loss"].append(float(avg_decoder_norm_loss))
        history["sparsity_loss"].append(float(avg_sparsity_loss))

        # Early stopping
        if avg_main_loss < best_loss - config["early_stop_delta"]:
            best_loss = avg_main_loss
            patience = 0
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                log_info(f"Early stopping triggered (patience={patience})")
                break

        del dataset, loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    
    log_info("=" * 60)
    log_info("FINAL DIAGNOSTICS")
    log_info("=" * 60)

    if detailed_stats:
        final_neuron_stats = compute_detailed_neuron_stats(sae, dead_neuron_threshold)
        log_detailed_neuron_stats(final_neuron_stats, ep + 1)

    if diagnose_per_dim and diag_data_cache is not None:
        final_dim_stats = compute_per_dim_reconstruction_stats(
            sae, diag_data_cache, config
        )
        log_per_dim_stats(final_dim_stats, ep + 1)

        np.savez(
            os.path.join(run_dir, "per_dim_diagnostics.npz"),
            per_dim_mse=final_dim_stats["per_dim_mse"],
            per_dim_var=final_dim_stats["per_dim_var"],
            per_dim_ev=final_dim_stats["per_dim_ev"],
            worst_dims=final_dim_stats["worst_dims"],
            worst_ev=final_dim_stats["worst_ev"],
        )

    
    final_act_dist = sae.get_activation_distribution()
    log_info(f"Final activation distribution:", prefix="ACTIV")
    log_info(
        f"  Main-only neurons: {final_act_dist['main_only_neurons']}", prefix="ACTIV"
    )
    log_info(
        f"  Aux-only neurons:  {final_act_dist['aux_only_neurons']}", prefix="ACTIV"
    )
    log_info(f"  Both (main+aux):   {final_act_dist['both_neurons']}", prefix="ACTIV")
    log_info(
        f"  Never activated:   {final_act_dist['never_activated']}", prefix="ACTIV"
    )

    sae.normalize_decoder()

    
    sd = sae._orig_mod.state_dict() if hasattr(sae, "_orig_mod") else sae.state_dict()
    torch.save(sd, model_path)
    log_info(f"Model saved to {model_path}")

    
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    
    final_stats_save = {
        "final_main_loss": float(avg_main_loss),
        "final_aux_loss": float(avg_aux_loss),
        "final_explained_var": float(avg_explained_var),
        "final_dead_ratio": float(dead_ratio_combined),
        "final_dead_ratio_main_only": float(dead_ratio_main),
        "final_never_ratio": float(never_ratio),
        "final_rare_ratio": float(rare_ratio),
        "final_max_inactive": float(max_inactive),
        "total_steps": int(global_step),
        "epochs_trained": int(ep + 1),
        
        "final_activation_distribution": {
            "main_only_neurons": int(final_act_dist["main_only_neurons"]),
            "aux_only_neurons": int(final_act_dist["aux_only_neurons"]),
            "both_neurons": int(final_act_dist["both_neurons"]),
            "never_activated": int(final_act_dist["never_activated"]),
            
            "main_rates_stats": {
                "mean": float(final_act_dist["main_rates"].mean()),
                "std": float(final_act_dist["main_rates"].std()),
                "max": float(final_act_dist["main_rates"].max()),
            },
            "aux_rates_stats": {
                "mean": float(final_act_dist["aux_rates"].mean()),
                "std": float(final_act_dist["aux_rates"].std()),
                "max": float(final_act_dist["aux_rates"].max()),
            },
        },
        "config_used": {
            "loss_type": loss_type,
            "huber_delta": huber_delta,
            "aux_k": aux_k,
            "sparsity_coeff": sparsity_coeff,
            "sparsity_target": sparsity_target,
            "decoder_norm_coeff": decoder_norm_coeff,
            "clip_extreme": config.get("clip_extreme_values", False),
            "enc_init_scale": enc_init_scale,
            "dead_neuron_threshold": dead_neuron_threshold,
        },
    }

    if detailed_stats:
        
        stats_to_save = {}
        for k, v in final_neuron_stats.items():
            if isinstance(v, np.ndarray):
                stats_to_save[k] = v.tolist()
            elif isinstance(v, dict):
                stats_to_save[k] = {
                    kk: (
                        vv.tolist()
                        if isinstance(vv, np.ndarray)
                        else (
                            int(vv)
                            if isinstance(vv, (np.integer, np.int32, np.int64))
                            else (
                                float(vv)
                                if isinstance(vv, (np.floating, np.float32, np.float64))
                                else vv
                            )
                        )
                    )
                    for kk, vv in v.items()
                }
            else:
                stats_to_save[k] = v
        final_stats_save["final_neuron_stats"] = stats_to_save

    if diagnose_per_dim:
        final_stats_save["final_dim_stats"] = {
            "total_ev": float(final_dim_stats["total_ev"]),
            "median_dim_ev": float(final_dim_stats["median_dim_ev"]),
            "dims_below_50pct_ev": int(final_dim_stats["dims_below_50pct_ev"]),
        }

    stats_path = os.path.join(run_dir, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(final_stats_save, f, indent=2)

    return model_path


# ==========================================

# ==========================================
def inference_token_level(run_dir, model_path, d_dict, k, preprocess_config, config):
    """Token-level inference with memory-efficient streaming"""
    log_info(f"--- Token-Level Inference D={d_dict} K={k} ---")

    out_path = os.path.join(run_dir, "sae_feats_token_csr.npz")
    mapping_path = os.path.join(run_dir, "token_to_seq_mapping.npz")

    if os.path.exists(out_path):
        log_info(f"Token features already exist: {out_path}", prefix="SKIP")
        return out_path

    device = get_device()
    use_amp = config["use_amp"]
    d_in = preprocess_config["d_in"]
    enc_init_scale = float(config.get("enc_init_scale", 0.1))

    
    sae = TopKSAE(d_in, d_dict, k, enc_init_scale=enc_init_scale).to(device)
    sae.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    sae.eval()

    
    stats_path = os.path.join(config["output_root"], "global_stats.npz")
    if os.path.exists(stats_path):
        stats = dict(np.load(stats_path))
        with torch.no_grad():
            if config.get("do_mean_center", True):
                center = stats.get("center", stats.get("mu"))
                sae.b_pre.copy_(torch.from_numpy(center).to(device))
            if config.get("do_rescale", True):
                sae.alpha.copy_(torch.from_numpy(stats["alpha"]).to(device))
            if config.get("clip_extreme_values", False) and "clip_min" in stats:
                center = stats.get("center", stats.get("mu"))
                alpha = stats["alpha"]
                raw_clip_min = stats["clip_min"]
                raw_clip_max = stats["clip_max"]
                norm_clip_min = ((raw_clip_min - center) * alpha).astype(np.float32)
                norm_clip_max = ((raw_clip_max - center) * alpha).astype(np.float32)
                sae.set_clip_bounds(norm_clip_min, norm_clip_max)
                log_info(
                    f"Restored clip bounds: [{norm_clip_min.min():.2f}, {norm_clip_max.max():.2f}]"
                )

    
    orig_tokens_path = preprocess_config.get("input_tokens_path")
    if not orig_tokens_path or not os.path.exists(orig_tokens_path):
        log_info("ERROR: Cannot locate token data", prefix="ERROR")
        return None

    offsets_path = None
    base_dir = os.path.dirname(orig_tokens_path)
    for off_p in [
        os.path.join(base_dir, "seq_token_offsets.npy"),
        os.path.join(os.path.dirname(base_dir), "seq_token_offsets.npy"),
        preprocess_config.get("staged_offsets_path", ""),
    ]:
        if off_p and os.path.exists(off_p):
            offsets_path = off_p
            break

    if not offsets_path:
        log_info("ERROR: Cannot locate offsets", prefix="ERROR")
        return None

    log_info(f"Loading token data: {orig_tokens_path}")
    X_mmap = np.load(orig_tokens_path, mmap_mode="r")
    offsets = np.load(offsets_path)

    n_total_tokens = X_mmap.shape[0]
    n_seqs = len(offsets) - 1
    log_info(f"Total tokens: {n_total_tokens:,}, Sequences: {n_seqs:,}")

    
    chunk_size = config.get("infer_chunk_size", 4096)
    all_indices = []
    all_values = []
    all_indptr = [0]
    token_seq_map = []

    current_nnz = 0
    processed_tokens = 0

    log_info(f"Processing in chunks of {chunk_size}...")

    with torch.no_grad():
        for start_idx in range(0, n_total_tokens, chunk_size):
            end_idx = min(start_idx + chunk_size, n_total_tokens)
            batch = (
                torch.from_numpy(np.array(X_mmap[start_idx:end_idx])).float().to(device)
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                topk_idx, topk_vals = sae.encode(batch)

            
            idx_np = topk_idx.cpu().numpy()
            vals_np = topk_vals.cpu().numpy()

            
            for i in range(len(batch)):
                row_idx = idx_np[i]
                row_vals = vals_np[i]

                
                mask = row_vals > 0
                row_idx = row_idx[mask]
                row_vals = row_vals[mask]

                all_indices.extend(row_idx.tolist())
                all_values.extend(row_vals.tolist())
                current_nnz += len(row_idx)
                all_indptr.append(current_nnz)

                
                token_global_idx = start_idx + i
                seq_id = np.searchsorted(offsets[1:], token_global_idx, side="right")
                token_seq_map.append(seq_id)

            processed_tokens += len(batch)

            if (start_idx // chunk_size) % 100 == 0:
                log_info(
                    f"  Processed {processed_tokens:,}/{n_total_tokens:,} tokens ({processed_tokens / n_total_tokens * 100:.1f}%)"
                )

            del batch, topk_idx, topk_vals
            if device.type == "cuda":
                torch.cuda.empty_cache()

    
    log_info("Building final CSR matrix...")
    indices = np.array(all_indices, dtype=np.int32)
    values = np.array(all_values, dtype=np.float32)
    indptr = np.array(all_indptr, dtype=np.int64)

    X_csr = sp.csr_matrix(
        (values, indices, indptr), shape=(n_total_tokens, d_dict), dtype=np.float32
    )

    log_info(
        f"Saving features: {X_csr.shape}, {X_csr.nnz:,} non-zeros ({X_csr.nnz / X_csr.shape[0] / X_csr.shape[1] * 100:.2f}% sparse)"
    )
    sp.save_npz(out_path, X_csr)

    
    np.savez(
        mapping_path,
        token_to_seq=np.array(token_seq_map, dtype=np.int32),
        seq_offsets=offsets,
    )

    log_info(f"Saved: {out_path}")
    return out_path


# ==========================================

# ==========================================
def eval_probe(run_dir, feats_path, config):
    log_info(f"--- Running Evaluation: {os.path.basename(run_dir)} ---")
    csv_path = os.path.join(run_dir, "eval_labels.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    X = sp.load_npz(feats_path)

    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    metrics = {}
    label_cols = ["Kingdom", "Phylum", "Class", "Order", "Family"]

    for col in label_cols:
        if col not in df.columns:
            continue
        y_raw = df[col].astype(str)
        mask = (
            (~y_raw.str.contains("Unclassified", case=False))
            & (y_raw.notna())
            & (y_raw != "nan")
        )
        valid_idx = np.where(mask)[0]
        if len(valid_idx) < 100:
            continue
        X_sub = X[valid_idx]
        y_sub = y_raw.iloc[valid_idx].values

        vc = pd.Series(y_sub).value_counts()
        keep = vc[vc >= 10].index
        mask2 = pd.Series(y_sub).isin(keep)
        X_fin = X_sub[mask2.values]
        y_fin = y_sub[mask2.values]
        if len(y_fin) < 50:
            continue

        le = LabelEncoder()
        y_enc = le.fit_transform(y_fin)
        Xt, Xv, yt, yv = train_test_split(
            X_fin, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        clf = SGDClassifier(loss="log_loss", max_iter=500, n_jobs=-1, random_state=42)
        clf.fit(Xt, yt)
        score = f1_score(yv, clf.predict(Xv), average="weighted")
        metrics[f"{col}_f1"] = float(score)
        print(f"    {col}: {score:.4f}")

    return metrics


# ==========================================

# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE Hybrid Pipeline (Optimized v2.4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocess_config.json (REQUIRED)",
    )
    parser.add_argument(
        "--labels_csv", type=str, default="", help="Path to labels CSV for evaluation"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./SAE_Hybrid_Results",
        help="Output directory",
    )
    parser.add_argument("--config", type=str, default=None, help="JSON config file")

    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=None,
        help="Dictionary dimensions to sweep",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=None, help="Top-K values to sweep"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")

    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["mse", "huber", "smooth_l1"],
        help="Loss function type",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=None, help="Delta for Huber/Smooth L1 loss"
    )
    parser.add_argument(
        "--sparsity_coeff", type=float, default=None, help="L1 sparsity coefficient"
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "train_only", "infer_only"],
        help="Which stages to run",
    )

    parser.add_argument(
        "--init_method",
        type=str,
        default="random",
        choices=["random", "kmeans_pp", "high_norm"],
        help="Initialization method",
    )

    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument(
        "--no_clip", action="store_true", help="Disable extreme value clipping"
    )
    parser.add_argument(
        "--robust_scaling", action="store_true", help="Use IQR-based robust scaling"
    )

    parser.add_argument(
        "--dead_threshold", type=int, default=None, help="Dead neuron threshold (steps)"
    )

    return parser.parse_args()


# ==========================================

# ==========================================
def main():
    args = parse_args()

    if args.config:
        log_info(f"Loading config from: {args.config}")
        try:
            with open(args.config) as f:
                config = json.load(f)
        except Exception as e:
            log_info(f"Failed to load config: {e}", prefix="ERROR")
            config = DEFAULT_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()

    
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.labels_csv:
        config["labels_csv_path"] = args.labels_csv
    if args.output_root:
        config["output_root"] = args.output_root
    if args.dims:
        config["sweep_dims"] = args.dims
    if args.ks:
        config["sweep_ks"] = args.ks
    if args.lr is not None:
        config["lr"] = args.lr
    if args.init_method:
        config["init_method"] = args.init_method

    if args.loss_type:
        config["loss_type"] = args.loss_type
    if args.huber_delta is not None:
        config["huber_delta"] = args.huber_delta
    if args.sparsity_coeff is not None:
        config["sparsity_coeff"] = args.sparsity_coeff
    if args.dead_threshold is not None:
        config["dead_neuron_threshold"] = args.dead_threshold

    config["use_torch_compile"] = not args.no_compile
    config["use_amp"] = not args.no_amp
    config["clip_extreme_values"] = not args.no_clip
    config["use_robust_scaling"] = args.robust_scaling

    if not config.get("data_dir"):
        log_info("ERROR: --data_dir is required!", prefix="ERROR")
        sys.exit(1)

    if not config.get("output_root") or str(config["output_root"]).strip() == "":
        config["output_root"] = "./SAE_Hybrid_Results"
        log_info(f"Using default output_root: {config['output_root']}", prefix="WARN")

    if not config.get("sweep_dims"):
        config["sweep_dims"] = [32768]
    if not config.get("sweep_ks"):
        config["sweep_ks"] = [64]

    pre_cfg_path = os.path.join(config["data_dir"], "preprocess_config.json")
    if not os.path.exists(pre_cfg_path):
        log_info(
            f"ERROR: preprocess_config.json not found at {pre_cfg_path}", prefix="ERROR"
        )
        sys.exit(1)

    with open(pre_cfg_path) as f:
        preprocess_config = json.load(f)

    if args.epochs is not None:
        preprocess_config["n_epochs"] = args.epochs

    log_info("=" * 60)
    log_info("SAE Hybrid Pipeline (Optimized v2.4)")
    log_info("=" * 60)
    log_info(f"Data dir:        {config['data_dir']}")
    log_info(f"Output root:     {config['output_root']}")
    log_info(f"Dimensions:      {config['sweep_dims']}")
    log_info(f"Top-K values:    {config['sweep_ks']}")
    log_info(f"Loss type:       {config.get('loss_type', 'huber')}")
    log_info(f"Huber delta:     {config.get('huber_delta', 10.0)}")
    log_info(f"Sparsity coeff:  {config.get('sparsity_coeff', 0.0)}")
    log_info(f"Dead threshold:  {config.get('dead_neuron_threshold', 1000)}")
    log_info(f"Clip extreme:    {config.get('clip_extreme_values', True)}")
    log_info(f"Robust scaling:  {config.get('use_robust_scaling', False)}")
    log_info(f"Init method:     {config.get('init_method', 'kmeans_pp')}")
    log_info(f"Enc init scale:  {config.get('enc_init_scale', 0.1)}")
    log_info(f"AMP:             {config['use_amp']}")
    log_info(f"Compile:         {config['use_torch_compile']}")
    log_info("=" * 60)

    
    stager = DataStager(config, preprocess_config)
    preprocess_config = stager.setup()

    
    ensure_dir(config["output_root"])
    stats_save_path = os.path.join(config["output_root"], "global_stats.npz")

    if args.stage in ["all", "train_only"]:
        
        log_info("Computing statistics for training...")
        stats = estimate_stats_from_chunks(preprocess_config, config)
        np.savez(stats_save_path, **stats)
        log_info(f"Statistics saved to: {stats_save_path}")
    elif args.stage == "infer_only":
        
        if os.path.exists(stats_save_path):
            log_info(f"Loading existing statistics from: {stats_save_path}")
            stats = dict(np.load(stats_save_path))
        else:
            log_info(
                "WARNING: Statistics file not found, computing now...", prefix="WARN"
            )
            stats = estimate_stats_from_chunks(preprocess_config, config)
            np.savez(stats_save_path, **stats)
    else:
        stats = {}

    summary = []
    tasks = []
    for d in config["sweep_dims"]:
        for k in config["sweep_ks"]:
            run_name = f"SAE_Tok_D{d}_K{k}"
            run_dir = os.path.join(config["output_root"], run_name)
            ensure_dir(run_dir)
            tasks.append(
                {"d": int(d), "k": int(k), "run_dir": run_dir, "run_name": run_name}
            )

    
    if args.stage in ["all", "train_only"]:
        log_info("=" * 60)
        log_info("PHASE 1: TRAINING ALL MODELS")
        log_info("=" * 60)
        for task in tasks:
            try:
                train_one_run(
                    task["run_dir"],
                    task["d"],
                    task["k"],
                    preprocess_config,
                    stats,
                    config,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"!!! Train failed for {task['run_name']}: {e}")
                import traceback

                traceback.print_exc()

    
    if args.stage in ["all", "infer_only"]:
        log_info("=" * 60)
        log_info("PHASE 2: INFERENCE & EVALUATION")
        log_info("=" * 60)
        for task in tasks:
            run_dir = task["run_dir"]
            model_path = os.path.join(run_dir, "sae_model.pt")
            if os.path.exists(model_path):
                try:
                    
                    feats_path = inference_token_level(  
                        run_dir,
                        model_path,
                        task["d"],
                        task["k"],
                        preprocess_config,
                        config,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if (
                        feats_path
                        and os.path.exists(feats_path)
                        and config.get("labels_csv_path")
                    ):
                        res = eval_probe(run_dir, feats_path, config)
                        res.update({"dim": task["d"], "k": task["k"]})
                        summary.append(res)
                        pd.DataFrame(summary).to_csv(
                            os.path.join(config["output_root"], "summary.csv"),
                            index=False,
                        )
                except Exception as e:
                    print(f"!!! Inference failed for {task['run_name']}: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                log_info(f"Model missing: {task['run_name']}")

    if summary:
        log_info("=" * 60)
        log_info("FINAL SUMMARY")
        log_info("=" * 60)
        df_summary = pd.DataFrame(summary)
        print(df_summary.to_string())

    log_info("All Done.")


if __name__ == "__main__":
    main()
