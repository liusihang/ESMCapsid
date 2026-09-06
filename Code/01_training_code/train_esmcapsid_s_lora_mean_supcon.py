#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ESMCapsid-S with LoRA and supervised contrastive learning.

The script accepts either explicit training and validation CSV files or a single
CSV for an internal split, uses masked mean pooling, and exports the merged model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from ml_dtypes import bfloat16  # noqa: F401
except Exception:
    bfloat16 = None  # noqa: F841
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from experiment_tools.vicapsid_training_schedule import build_step_eval_schedule
except ImportError:
    from vicapsid_training_schedule import build_step_eval_schedule

try:
    from peft import LoraConfig, get_peft_model

    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


class SeqClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 786):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        assert {"seq", "label"}.issubset(self.df.columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        seq = str(self.df.loc[i, "seq"])[: self.max_len]
        label = float(self.df.loc[i, "label"])
        enc = self.tok(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        return item


def _to_float32_array_safe(x):
    import numpy as _np
    import torch as _torch

    if isinstance(x, (list, tuple)):
        cands = []
        for e in x:
            try:
                cands.append(_to_float32_array_safe(e))
            except Exception:
                pass
        for arr in cands:
            if (
                isinstance(arr, _np.ndarray)
                and arr.ndim == 2
                and arr.shape[0] >= arr.shape[1]
            ):
                return arr
        for arr in cands:
            if isinstance(arr, _np.ndarray):
                return arr
        return _to_float32_array_safe(x[0])
    if isinstance(x, _torch.Tensor):
        return x.detach().to(_torch.float32).cpu().numpy()
    arr = _np.asarray(x)
    if (
        hasattr(arr.dtype, "name") and arr.dtype.name == "bfloat16"
    ) or arr.dtype == _np.float16:
        arr = arr.astype(_np.float32, copy=False)
    if arr.dtype not in (_np.float32, _np.float64):
        arr = arr.astype(_np.float32, copy=False)
    return arr


def _normalize_logits_shape(logits, n_labels_hint: Optional[int] = None):
    if logits.ndim == 1:
        return logits.reshape(-1, 1)
    if logits.ndim == 2:
        rows, cols = logits.shape
        if cols > 0 and rows <= 8 and cols > rows:
            logits = logits.T
        elif (
            n_labels_hint is not None
            and rows == n_labels_hint
            and cols != n_labels_hint
        ):
            logits = logits.T
        return logits
    return logits.reshape(logits.shape[0], -1)


def compute_metrics_fn(eval_pred):
    try:
        from transformers.trainer_utils import EvalPrediction

        if isinstance(eval_pred, EvalPrediction):
            raw_logits, labels = eval_pred.predictions, eval_pred.label_ids
        else:
            raw_logits, labels = eval_pred
    except Exception:
        raw_logits, labels = eval_pred

    logits = _to_float32_array_safe(raw_logits)
    labels = _to_float32_array_safe(labels).reshape(-1).astype(int)
    logits = _normalize_logits_shape(logits)

    z = np.clip(logits[:, 0], -50, 50)
    probs = 1.0 / (1.0 + np.exp(-z))
    pred05 = (probs >= 0.5).astype(int)

    ap = average_precision_score(labels, probs)
    try:
        roc = roc_auc_score(labels, probs)
    except Exception:
        roc = float("nan")

    f1_05 = f1_score(labels, pred05) if len(np.unique(labels)) > 1 else 0.0
    mcc = matthews_corrcoef(labels, pred05) if len(np.unique(labels)) > 1 else 0.0
    bal = balanced_accuracy_score(labels, pred05) if len(np.unique(labels)) > 1 else 0.0

    prec, rec, thr = precision_recall_curve(labels, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    thr_best = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    f1_best = float(np.nanmax(f1s))

    return {
        "average_precision": float(ap),
        "roc_auc": float(roc),
        "mcc@0.5": float(mcc),
        "bal_acc@0.5": float(bal),
        "f1@0.5": float(f1_05),
        "f1_best": float(f1_best),
        "thr_best": float(thr_best),
    }


def masked_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if features.shape[0] < 2:
        return features.new_zeros(())

    labels = labels.view(-1)
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return features.new_zeros(())

    exp_logits = torch.exp(logits) * (~self_mask).to(dtype=logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_mask_f = positive_mask.to(dtype=log_prob.dtype)
    mean_log_prob_pos = (positive_mask_f * log_prob).sum(dim=1) / positive_mask_f.sum(
        dim=1
    ).clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


class BalancedBinaryBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        seed: int = 42,
        positive_fraction: float = 0.5,
        min_per_class: int = 2,
    ):
        if batch_size < 4:
            raise ValueError(
                "batch_size must be at least 4 for balanced binary sampling"
            )
        self.labels = [int(x) for x in labels]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.positive_fraction = float(positive_fraction)
        self.min_per_class = int(min_per_class)
        self.epoch = 0
        self.pos_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.neg_indices = [i for i, label in enumerate(self.labels) if label == 0]
        if not self.pos_indices or not self.neg_indices:
            raise ValueError(
                "balanced binary sampling requires both positive and negative samples"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.labels) / self.batch_size))

    def _draw(
        self,
        pool: List[int],
        cursor: int,
        need: int,
        rng: random.Random,
    ) -> Tuple[List[int], int]:
        drawn: List[int] = []
        if need <= 0:
            return drawn, cursor
        while len(drawn) < need:
            remaining = len(pool) - cursor
            take = min(need - len(drawn), remaining)
            drawn.extend(pool[cursor : cursor + take])
            cursor += take
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
        return drawn, cursor

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pos_pool = self.pos_indices[:]
        neg_pool = self.neg_indices[:]
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)
        pos_cursor = 0
        neg_cursor = 0

        target_pos = int(round(self.batch_size * self.positive_fraction))
        target_pos = max(self.min_per_class, target_pos)
        target_pos = min(target_pos, self.batch_size - self.min_per_class)
        target_neg = self.batch_size - target_pos

        for _ in range(len(self)):
            pos_batch, pos_cursor = self._draw(pos_pool, pos_cursor, target_pos, rng)
            neg_batch, neg_cursor = self._draw(neg_pool, neg_cursor, target_neg, rng)
            batch = pos_batch + neg_batch
            rng.shuffle(batch)
            yield batch


class LearningCurveCallback(TrainerCallback):
    def __init__(self, outdir: str, ema_beta: float = 0.9):
        self.outdir = outdir
        self.ema_beta = float(ema_beta)
        os.makedirs(self.outdir, exist_ok=True)
        self.train_log, self.eval_log = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if "loss" in logs:
            self.train_log.append(
                {
                    "step": int(state.global_step),
                    "epoch": float(logs.get("epoch", state.epoch or 0.0)),
                    "loss": float(logs["loss"]),
                }
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        metrics = metrics or {}
        rec = {"epoch": float(state.epoch or 0.0)}
        for k, v in metrics.items():
            if k.startswith("eval_") and isinstance(v, (int, float)):
                rec[k] = float(v)
        self.eval_log.append(rec)
        self._save_curves()

    def on_train_end(self, args, state, control, **kwargs):
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        self._save_curves(final=True)

    def _save_curves(self, final: bool = False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.train_log:
            df_tr = pd.DataFrame(self.train_log).sort_values("step")
            df_tr.to_csv(
                os.path.join(self.outdir, "learning_curve_train_steps.csv"), index=False
            )
        else:
            df_tr = None

        if self.eval_log:
            df_ev = pd.DataFrame(self.eval_log).sort_values("epoch")
            df_ev.to_csv(
                os.path.join(self.outdir, "learning_curve_eval_epochs.csv"), index=False
            )
        else:
            df_ev = None

        if df_tr is not None and len(df_tr) > 0:
            x = df_tr["step"].to_numpy()
            y = df_tr["loss"].to_numpy()
            beta = self.ema_beta
            ema = []
            running = None
            for value in y:
                running = (
                    value if running is None else beta * running + (1.0 - beta) * value
                )
                ema.append(running)
            ema = np.array(ema)

            plt.figure(figsize=(7, 4.2))
            plt.plot(x, y, lw=1.0, alpha=0.4, label="train loss")
            plt.plot(x, ema, lw=2.0, label=f"EMA (beta={beta})")
            plt.xlabel("global step")
            plt.ylabel("loss")
            plt.title("Training Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "curve_train_loss.png"), dpi=160)
            plt.close()

        if df_ev is not None and len(df_ev) > 0:
            for metric in [
                "eval_average_precision",
                "eval_roc_auc",
                "eval_f1@0.5",
                "eval_f1_best",
                "eval_mcc@0.5",
                "eval_bal_acc@0.5",
                "eval_loss",
            ]:
                if metric not in df_ev.columns:
                    continue
                plt.figure(figsize=(6.2, 4.0))
                plt.plot(df_ev["epoch"], df_ev[metric], marker="o")
                plt.xlabel("epoch")
                plt.ylabel(metric.replace("eval_", ""))
                plt.title(metric.replace("eval_", "").upper())
                plt.tight_layout()
                safe = metric.replace("eval_", "").replace("@", "at").replace("/", "_")
                plt.savefig(os.path.join(self.outdir, f"curve_{safe}.png"), dpi=160)
                plt.close()


_ALLOWED_LORA_TYPES = (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def resolve_lora_targets(
    model: nn.Module, substrings: Iterable[str]
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    exact: List[str] = []
    hit_stats: Dict[str, int] = {s: 0 for s in substrings}
    skipped_stats: Dict[str, int] = {s: 0 for s in substrings}
    seen = set()
    for name, mod in model.named_modules():
        low = name.lower()
        for sub in substrings:
            if sub and sub in low:
                if isinstance(mod, _ALLOWED_LORA_TYPES):
                    if name not in seen:
                        exact.append(name)
                        seen.add(name)
                    hit_stats[sub] += 1
                else:
                    skipped_stats[sub] += 1
    return exact, hit_stats, skipped_stats


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", help="CSV containing seq and label; used for an internal split when explicit train/validation files are not supplied"
    )
    ap.add_argument("--train-csv", help="Explicit training CSV containing seq and label")
    ap.add_argument("--val-csv", help="Explicit validation CSV containing seq and label")
    ap.add_argument(
        "--split-summary-json", help="Optional external split-summary JSON copied into the final metrics"
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Local model directory or Hugging Face identifier requiring trust_remote_code",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-len", type=int, default=786)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument(
        "--eval-steps", type=int, default=0, help="Use step-based evaluation and checkpointing when greater than zero"
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Enable EarlyStoppingCallback when greater than zero",
    )
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-drop", type=float, default=0.01)
    ap.add_argument(
        "--target-modules",
        nargs="+",
        default=["layernorm_qkv.1", "out_proj", "fc1", "fc2", "dense"],
        help="Substrings used to select LoRA target modules",
    )
    ap.add_argument(
        "--modules-to-save",
        nargs="+",
        default=[
            "mean_classifier",
            "mean_classifier_dropout",
            "contrastive_projection",
        ],
        help="Modules retained as trainable and saved with the adapter",
    )
    ap.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--projection-dim", type=int, default=128)
    ap.add_argument("--lambda-supcon", type=float, default=0.05)
    ap.add_argument("--supcon-temperature", type=float, default=0.07)
    ap.add_argument("--sampler-positive-fraction", type=float, default=0.5)
    ap.add_argument("--dump-val-embeddings", action="store_true")
    return ap.parse_args()


def _load_split_frames(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata: dict[str, object] = {}

    if args.train_csv or args.val_csv:
        if not (args.train_csv and args.val_csv):
            raise ValueError("Explicit split mode requires both --train-csv and --val-csv")

        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
        metadata["split_mode"] = "explicit_csvs"
        metadata["train_csv"] = args.train_csv
        metadata["val_csv"] = args.val_csv
    else:
        if not args.csv:
            raise ValueError("Provide --csv or both --train-csv and --val-csv")
        df = pd.read_csv(args.csv)
        if not {"seq", "label"}.issubset(df.columns):
            raise ValueError("CSV must contain seq and label columns")
        df = df.dropna(subset=["seq", "label"]).copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0)
        df["label"] = (df["label"] > 0).astype(float)
        train_df, val_df = train_test_split(
            df,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=df["label"],
        )
        metadata["split_mode"] = "internal_stratified_split"
        metadata["source_csv"] = args.csv
        metadata["val_size"] = float(args.val_size)

    for frame_name, frame in [("train", train_df), ("validation", val_df)]:
        if not {"seq", "label"}.issubset(frame.columns):
            raise ValueError(f"{frame_name} CSV must contain seq and label columns")

    train_df = train_df.dropna(subset=["seq", "label"]).copy()
    val_df = val_df.dropna(subset=["seq", "label"]).copy()
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0)
    train_df["label"] = (train_df["label"] > 0).astype(float)
    val_df["label"] = pd.to_numeric(val_df["label"], errors="coerce").fillna(0)
    val_df["label"] = (val_df["label"] > 0).astype(float)

    if args.split_summary_json:
        metadata["split_summary_json"] = args.split_summary_json
        with open(args.split_summary_json, "r", encoding="utf-8") as handle:
            metadata["split_summary"] = json.load(handle)

    return train_df, val_df, metadata


def _get_train_modules(model: nn.Module) -> tuple[nn.Module, nn.Module, nn.Module]:
    if hasattr(model, "module"):
        model = model.module
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    return (
        base_model.mean_classifier_dropout,
        base_model.mean_classifier,
        base_model.contrastive_projection,
    )


def forward_mean_supcon(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model forward did not return hidden_states")
    pooled = masked_mean_pool(hidden_states[-1], inputs["attention_mask"])
    classifier_dropout, mean_classifier, contrastive_projection = _get_train_modules(
        model
    )
    logits = mean_classifier(classifier_dropout(pooled)).squeeze(-1)
    projected = contrastive_projection(pooled)
    return {
        "logits": logits,
        "pooled_embeddings": pooled,
        "projected_embeddings": projected,
    }


class MeanSupConTrainer(Trainer):
    def __init__(
        self,
        *args,
        pos_weight: float = 1.0,
        lambda_supcon: float = 0.05,
        supcon_temperature: float = 0.07,
        sampler_positive_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pos_weight_value = float(pos_weight)
        self.lambda_supcon = float(lambda_supcon)
        self.supcon_temperature = float(supcon_temperature)
        self.sampler_positive_fraction = float(sampler_positive_fraction)
        self._train_batch_sampler: Optional[BalancedBinaryBatchSampler] = None

    def _compute_loss_components(self, model, inputs):
        labels = inputs["labels"].float()
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = forward_mean_supcon(model, model_inputs)
        logits = outputs["logits"]
        pos_weight_tensor = torch.tensor(
            [self.pos_weight_value], device=logits.device, dtype=logits.dtype
        )
        cls_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels.to(dtype=logits.dtype),
            pos_weight=pos_weight_tensor,
        )
        supcon_loss = supervised_contrastive_loss(
            outputs["projected_embeddings"],
            labels.to(dtype=torch.long),
            temperature=self.supcon_temperature,
        )
        total_loss = cls_loss + self.lambda_supcon * supcon_loss
        return total_loss, cls_loss.detach(), supcon_loss.detach(), outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        total_loss, cls_loss, supcon_loss, outputs = self._compute_loss_components(
            model, inputs
        )
        if return_outputs:
            return (
                total_loss,
                {
                    "logits": outputs["logits"].unsqueeze(-1),
                    "cls_loss": cls_loss,
                    "supcon_loss": supcon_loss,
                },
            )
        return total_loss

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            total_loss, _, _, outputs = self._compute_loss_components(model, inputs)
        logits = outputs["logits"].unsqueeze(-1)
        labels = inputs["labels"]
        if prediction_loss_only:
            return (total_loss.detach(), None, None)
        return (total_loss.detach(), logits.detach(), labels.detach())

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        labels = [int(x) for x in self.train_dataset.df["label"].tolist()]
        sampler = BalancedBinaryBatchSampler(
            labels=labels,
            batch_size=self.args.train_batch_size,
            seed=self.args.seed,
            positive_fraction=self.sampler_positive_fraction,
        )
        self._train_batch_sampler = sampler
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def save_validation_outputs(self, outdir: str, val_df: pd.DataFrame):
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        model.eval()
        device = next(model.parameters()).device

        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        pooled_chunks = []
        logits_chunks = []
        labels_chunks = []

        for batch in loader:
            labels = batch["labels"]
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = forward_mean_supcon(model, model_inputs)
            pooled_chunks.append(
                outputs["pooled_embeddings"].detach().to(torch.float32).cpu().numpy()
            )
            logits_chunks.append(
                outputs["logits"].detach().to(torch.float32).cpu().numpy()
            )
            labels_chunks.append(labels.detach().to(torch.float32).cpu().numpy())

        pooled = np.concatenate(pooled_chunks, axis=0)
        logits = np.concatenate(logits_chunks, axis=0)
        labels = np.concatenate(labels_chunks, axis=0)

        np.save(os.path.join(outdir, "validation_mean_embeddings.npy"), pooled)
        np.save(os.path.join(outdir, "validation_logits.npy"), logits)
        np.save(os.path.join(outdir, "validation_labels.npy"), labels)
        val_df.reset_index(drop=True).to_csv(
            os.path.join(outdir, "validation_metadata.csv"),
            index=False,
        )


def main():
    args = build_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    tr, va, split_metadata = _load_split_frames(args)
    npos = int((tr["label"] == 1).sum())
    nneg = int((tr["label"] == 0).sum())
    pos_weight = float(max(1.0, nneg / max(1, npos)))
    print(
        f"[info] split_mode={split_metadata['split_mode']} train={len(tr)} val={len(va)} "
        f"| pos_weight={pos_weight:.3f}"
    )

    cfg = AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    cfg.num_labels = 1
    cfg.problem_type = "single_label_classification"
    cfg.output_hidden_states = True
    cfg.output_attentions = False
    cfg.return_dict = True
    cfg.pooling_mode = "mean"
    cfg.projection_dim = int(args.projection_dim)
    cfg.lambda_supcon = float(args.lambda_supcon)
    cfg.supcon_temperature = float(args.supcon_temperature)
    cfg.training_head = "training_only_mean_classifier"

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        config=cfg,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        ignore_mismatched_sizes=True,
    )
    dropout_prob = float(
        getattr(cfg, "classifier_dropout", None)
        or getattr(cfg, "hidden_dropout_prob", 0.1)
        or 0.1
    )
    base.mean_classifier_dropout = nn.Dropout(dropout_prob)
    base.mean_classifier = nn.Linear(int(cfg.hidden_size), 1)
    base.contrastive_projection = ProjectionHead(
        int(cfg.hidden_size), int(args.projection_dim)
    )

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    except Exception:
        tok = getattr(base, "tokenizer", None)
    assert (
        tok is not None
    ), "Tokenizer was not available; ensure trust_remote_code=True and that the model package contains tokenizer files."
    print("[info] tokenizer ready")

    exact_targets, hit_stats, skipped_stats = resolve_lora_targets(
        base, [s.lower() for s in args.target_modules]
    )
    if not exact_targets:
        fallback = ["layernorm_qkv.1", "out_proj"]
        exact_targets, _, _ = resolve_lora_targets(base, [s.lower() for s in fallback])
        print(
            f"[warn] target-module substrings matched no supported layer; using fallback {fallback} -> matched {len(exact_targets)} layers"
        )
        if not exact_targets:
            raise RuntimeError(
                "No supported LoRA target could be resolved; check model module names and --target-modules."
            )

    is_rank0 = os.environ.get("RANK", "0") == "0"
    if is_rank0:
        with open(
            os.path.join(args.outdir, "lora_targets.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "exact_targets": exact_targets,
                    "modules_to_save": args.modules_to_save,
                    "hit_stats": hit_stats,
                    "skipped_stats": skipped_stats,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    if not HAS_PEFT:
        raise RuntimeError("PEFT is required for LoRA training.")
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        bias="none",
        target_modules=exact_targets,
        modules_to_save=args.modules_to_save,
    )
    model = get_peft_model(base, lcfg)
    if is_rank0:
        model.print_trainable_parameters()

    max_len = min(args.max_len, getattr(tok, "model_max_length", args.max_len))
    ds_tr = SeqClsDataset(tr, tok, max_len=max_len)
    ds_va = SeqClsDataset(va, tok, max_len=max_len)

    use_cuda = torch.cuda.is_available()
    bs_eval = max(1, args.bs // 2)
    callback_list = []
    if args.eval_steps > 0:
        schedule_options = build_step_eval_schedule(
            eval_steps=args.eval_steps,
            early_stopping_patience=(
                args.early_stopping_patience
                if args.early_stopping_patience > 0
                else None
            ),
        )
        if args.early_stopping_patience > 0:
            callback_list.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            )
    else:
        schedule_options = {
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
        }

    targs = TrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=bs_eval,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy=schedule_options["evaluation_strategy"],
        save_strategy=schedule_options["save_strategy"],
        load_best_model_at_end=bool(schedule_options["load_best_model_at_end"]),
        metric_for_best_model="average_precision",
        greater_is_better=True,
        fp16=args.fp16 and use_cuda and not args.bf16,
        bf16=args.bf16 and use_cuda and not args.fp16,
        eval_accumulation_steps=8,
        dataloader_num_workers=args.num_workers,
        logging_strategy="steps",
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        ddp_backend="nccl" if use_cuda else None,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=max(1, args.grad_accum),
        remove_unused_columns=False,
        eval_steps=(
            int(schedule_options["eval_steps"])
            if "eval_steps" in schedule_options
            else None
        ),
        save_steps=(
            int(schedule_options["save_steps"])
            if "save_steps" in schedule_options
            else None
        ),
    )

    trainer = MeanSupConTrainer(
        model=model,
        args=targs,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        compute_metrics=compute_metrics_fn,
        pos_weight=pos_weight,
        lambda_supcon=args.lambda_supcon,
        supcon_temperature=args.supcon_temperature,
        sampler_positive_fraction=args.sampler_positive_fraction,
    )
    trainer.add_callback(LearningCurveCallback(args.outdir, ema_beta=0.9))
    for callback in callback_list:
        trainer.add_callback(callback)

    trainer.train()
    metrics = trainer.evaluate()

    if trainer.is_world_process_zero():
        print("[final eval]", json.dumps(metrics, indent=2))
        out = {
            k.replace("eval_", ""): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        thr_best = out.get("thr_best")

        if args.dump_val_embeddings:
            trainer.save_validation_outputs(args.outdir, va)

        model.save_pretrained(args.outdir)
        tok.save_pretrained(args.outdir)

        out.update(
            {
                "pos_weight": float(pos_weight),
                "train_rows": int(len(tr)),
                "validation_rows": int(len(va)),
                "train_positive_count": int(npos),
                "train_negative_count": int(nneg),
                "validation_positive_count": int((va["label"] == 1).sum()),
                "validation_negative_count": int((va["label"] == 0).sum()),
                "selected_per_device_batch_size": int(args.bs),
                "selected_gradient_accumulation": int(args.grad_accum),
                "split_mode": split_metadata["split_mode"],
                "eval_steps": int(args.eval_steps),
                "early_stopping_patience": int(args.early_stopping_patience),
                "pooling_mode": "mean",
                "projection_dim": int(args.projection_dim),
                "lambda_supcon": float(args.lambda_supcon),
                "supcon_temperature": float(args.supcon_temperature),
                "sampler_positive_fraction": float(args.sampler_positive_fraction),
                "training_head": "training_only_mean_classifier",
            }
        )
        if "split_summary_json" in split_metadata:
            out["split_summary_json"] = split_metadata["split_summary_json"]
        if "train_csv" in split_metadata:
            out["train_csv"] = split_metadata["train_csv"]
        if "val_csv" in split_metadata:
            out["val_csv"] = split_metadata["val_csv"]
        if "source_csv" in split_metadata:
            out["source_csv"] = split_metadata["source_csv"]
        if "split_summary" in split_metadata:
            out["split_summary"] = split_metadata["split_summary"]
        with open(
            os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(out, handle, indent=2)

        print(f"[ok] adapter+head saved to: {args.outdir}")
        if thr_best is not None:
            print(f"[hint] recommended deployment threshold (thr_best) {thr_best:.4f}")

        try:
            merged = model.merge_and_unload()
            merged_dir = os.path.join(args.outdir, "merged_full_model")
            os.makedirs(merged_dir, exist_ok=True)
            merged.save_pretrained(merged_dir)
            tok.save_pretrained(merged_dir)
            print(f"[ok] merged full model saved to: {merged_dir}")
        except Exception as exc:
            print(f"[warn] merge_and_unload failed; deploy the adapter instead: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
