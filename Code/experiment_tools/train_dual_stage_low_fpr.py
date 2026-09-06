from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

try:
    from experiment_tools.layer_selection_utils import (
        apply_cascade,
        build_binary_targets,
        compute_binary_metrics,
        false_positives_per_million,
        mean_std,
        select_best_row_by_max_fpr,
        select_best_row_by_max_fpr_max_precision,
        select_best_row_by_min_recall,
        select_best_row_by_min_recall_max_precision,
        split_inner_train_calibration,
        sweep_thresholds,
    )
    from experiment_tools.linear_probe_cv_gpu import apply_standardize, standardize_fit
    from experiment_tools.linear_probe_utils import build_stratified_folds
except ImportError:  # pragma: no cover - remote flat-script fallback
    from layer_selection_utils import (
        apply_cascade,
        build_binary_targets,
        compute_binary_metrics,
        false_positives_per_million,
        mean_std,
        select_best_row_by_max_fpr,
        select_best_row_by_max_fpr_max_precision,
        select_best_row_by_min_recall,
        select_best_row_by_min_recall_max_precision,
        split_inner_train_calibration,
        sweep_thresholds,
    )
    from linear_probe_cv_gpu import apply_standardize, standardize_fit
    from linear_probe_utils import build_stratified_folds


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_hidden_dims(
    hidden_dims: list[int] | None, hidden_dim: int | None
) -> list[int]:
    if hidden_dims:
        dims = [int(dim) for dim in hidden_dims if int(dim) > 0]
        if dims:
            return dims
    if hidden_dim is not None:
        return [int(hidden_dim)]
    return [512]


def choose_training_backend(device: str, cache_on_device: bool | None) -> str:
    if cache_on_device is False:
        return "dataloader"
    if str(device).startswith("cuda"):
        return "device_tensor"
    return "dataloader"


def build_binary_mlp(input_dim: int, hidden_dims: list[int]):
    from torch import nn

    dims = normalize_hidden_dims(hidden_dims=hidden_dims, hidden_dim=None)
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_size in dims:
        layers.extend(
            [
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            ]
        )
        prev_dim = hidden_size
    layers.append(nn.Linear(prev_dim, 2))
    return nn.Sequential(*layers)


def fit_mlp_classifier(
    x_train,
    y_train,
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    seed: int,
    cache_on_device: bool | None = None,
):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_binary_mlp(int(x_train.shape[1]), hidden_dims).to(device)
    negatives = max(sum(1 for label in y_train if label == 0), 1)
    positives = max(sum(1 for label in y_train if label == 1), 1)
    class_weights = torch.tensor(
        [len(y_train) / (2.0 * negatives), len(y_train) / (2.0 * positives)],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    backend = choose_training_backend(device=device, cache_on_device=cache_on_device)
    batch_size = min(batch_size, len(x_train))
    use_amp = (
        backend == "device_tensor"
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    if backend == "device_tensor":
        train_x = torch.from_numpy(x_train.astype(np.float32, copy=False)).to(
            device, non_blocking=True
        )
        train_y = torch.as_tensor(np.asarray(y_train, dtype=np.int64), device=device)
        loader = None
    else:
        dataset = TensorDataset(
            torch.from_numpy(x_train.astype(np.float32, copy=False)),
            torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=device.startswith("cuda"),
        )
        train_x = None
        train_y = None

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        if backend == "device_tensor":
            assert train_x is not None and train_y is not None
            perm = torch.randperm(train_y.shape[0], device=device)
            for start in range(0, int(train_y.shape[0]), batch_size):
                idx = perm[start : start + batch_size]
                batch_x = train_x.index_select(0, idx)
                batch_y = train_y.index_select(0, idx)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        logits = model(batch_x)
                        loss = loss_fn(logits, batch_y)
                else:
                    logits = model(batch_x)
                    loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_seen = int(batch_y.shape[0])
                epoch_loss += float(loss.item()) * batch_seen
                seen += batch_seen
        else:
            assert loader is not None
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_seen = int(batch_y.shape[0])
                epoch_loss += float(loss.item()) * batch_seen
                seen += batch_seen

        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / max(seen, 1)})

    return model, history


def predict_positive_prob(model, x_eval, device: str):
    import numpy as np
    import torch

    model.eval()
    with torch.no_grad():
        if isinstance(x_eval, torch.Tensor):
            eval_tensor = x_eval.to(device, non_blocking=True)
        else:
            eval_tensor = torch.from_numpy(x_eval.astype(np.float32, copy=False)).to(
                device, non_blocking=True
            )
        logits = model(eval_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs.tolist()


def build_stage2_train_indices(
    train_idx: list[int],
    labels: list[int],
    stage1_train_probs: list[float],
    stage1_threshold: float,
    min_hard_negatives: int,
) -> list[int]:
    positive_idx = [idx for idx in train_idx if labels[idx] == 1]
    negative_pairs = [
        (float(prob), idx)
        for prob, idx in zip(stage1_train_probs, train_idx)
        if labels[idx] == 0
    ]
    hard_negatives = [idx for prob, idx in negative_pairs if prob >= stage1_threshold]
    negative_pairs.sort(key=lambda item: item[0], reverse=True)

    target_negative_n = min(
        len(negative_pairs),
        max(min_hard_negatives, len(positive_idx)),
    )
    if len(hard_negatives) < target_negative_n:
        supplement = [idx for _, idx in negative_pairs[:target_negative_n]]
        hard_negatives = list(dict.fromkeys(hard_negatives + supplement))

    return sorted(set(positive_idx + hard_negatives))


def build_ultralowfpr_stage2_train_indices(
    train_idx: list[int],
    labels: list[int],
    stage1_train_probs: list[float],
    stage1_threshold: float,
    min_hard_negatives: int,
    negative_multiplier: float,
) -> list[int]:
    positive_idx = [idx for idx in train_idx if labels[idx] == 1]
    negative_pairs = [
        (float(prob), idx)
        for prob, idx in zip(stage1_train_probs, train_idx)
        if labels[idx] == 0
    ]
    hard_negatives = [idx for prob, idx in negative_pairs if prob >= stage1_threshold]
    negative_pairs.sort(key=lambda item: item[0], reverse=True)

    desired_negative_n = max(
        min_hard_negatives,
        int(round(len(positive_idx) * negative_multiplier)),
    )
    desired_negative_n = min(len(negative_pairs), desired_negative_n)

    if len(hard_negatives) < desired_negative_n:
        supplement = [idx for _, idx in negative_pairs[:desired_negative_n]]
        hard_negatives = list(dict.fromkeys(hard_negatives + supplement))
    else:
        hard_negatives = hard_negatives[:desired_negative_n]

    return sorted(set(positive_idx + hard_negatives))


def summarize_model_rows(rows: list[dict]) -> dict[str, float]:
    metrics = {}
    for key in (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "fpr",
        "f1",
        "balanced_accuracy",
    ):
        mean_value, std_value = mean_std([float(row[key]) for row in rows])
        metrics[f"{key}_mean"] = mean_value
        metrics[f"{key}_std"] = std_value
    return metrics


def select_stage1_threshold_row(
    rows: list[dict[str, float]],
    min_recall: float,
    selection_mode: str,
) -> dict[str, float]:
    if selection_mode == "max_precision":
        return select_best_row_by_min_recall_max_precision(rows, min_recall=min_recall)
    return select_best_row_by_min_recall(rows, min_recall=min_recall)


def select_stage2_threshold_row(
    rows: list[dict[str, float]],
    max_fpr: float,
    selection_mode: str,
) -> dict[str, float]:
    if selection_mode == "max_precision":
        return select_best_row_by_max_fpr_max_precision(rows, max_fpr=max_fpr)
    return select_best_row_by_max_fpr(rows, max_fpr=max_fpr)


def _sorted_predictions_df(df):
    sort_cols = ["fold"]
    if "prot_id" in df.columns:
        sort_cols.append("prot_id")
    return df.sort_values(sort_cols)


def build_summary(
    *,
    predictions_df,
    fold_metric_rows: list[dict],
    model_name: str,
    early_layer: int,
    late_layer: int,
    stage1_min_recall: float,
    stage1_selection_mode: str,
    stage2_max_fpr: float,
    stage2_selection_mode: str,
    stage2_mode: str,
    stage2_negative_multiplier: float,
):
    summary = {
        "model_name": model_name,
        "early_layer": early_layer,
        "late_layer": late_layer,
        "stage1_min_recall": stage1_min_recall,
        "stage1_selection_mode": stage1_selection_mode,
        "stage2_max_fpr": stage2_max_fpr,
        "stage2_selection_mode": stage2_selection_mode,
        "stage2_mode": stage2_mode,
        "stage2_negative_multiplier": stage2_negative_multiplier,
        "n_samples": int(len(predictions_df)),
        "n_positive": int(predictions_df["binary_target"].sum()),
        "n_negative": int((predictions_df["binary_target"] == 0).sum()),
        "models": {},
    }

    for current_model_name in ("early", "late", "cascade"):
        pred_col = f"{current_model_name}_pred"
        y_true = predictions_df["binary_target"].astype(int).tolist()
        y_pred = predictions_df[pred_col].astype(int).tolist()
        metrics = compute_binary_metrics(y_true, y_pred)
        metrics["false_positives_per_million"] = false_positives_per_million(
            fp=metrics["fp"],
            n_negative=metrics["fp"] + metrics["tn"],
        )
        current_fold_rows = [
            row for row in fold_metric_rows if row["model"] == current_model_name
        ]
        metrics.update(summarize_model_rows(current_fold_rows))
        summary["models"][current_model_name] = metrics
    return summary


def write_readme(
    output_root: Path,
    *,
    model_name: str,
    embedding_dir: str,
    metadata_path: str,
    selection_json: str,
    early_layer: int,
    late_layer: int,
    batch_size: int,
    parallel_folds: int,
    stage1_min_recall: float,
    stage1_selection_mode: str,
    stage2_max_fpr: float,
    stage2_selection_mode: str,
):
    readme = f"""# Dual-stage fixed-layer rerun

- model: `{model_name}`
- embedding_dir: `{embedding_dir}`
- metadata: `{metadata_path}`
- selection_json: `{selection_json}`
- early_layer: `{early_layer}` (mean pooled)
- late_layer: `{late_layer}` (mean pooled)
- batch_size: `{batch_size}`
- parallel_folds: `{parallel_folds}`
- stage1 rule: recall >= `{stage1_min_recall}` with `{stage1_selection_mode}`
- stage2 rule: FPR <= `{stage2_max_fpr}` with `{stage2_selection_mode}`

## Outputs

- `predictions.csv`: pooled out-of-fold predictions
- `fold_metrics.csv`: per-fold metrics for early / late / cascade
- `thresholds.csv`: per-fold selected thresholds
- `training_history.csv`: per-fold train loss history
- `summary.json`: pooled summary metrics
- `selected_layers_used.json`: recorded early/late layers
- `fold_runs/`: raw per-fold artifacts for the single-GPU parallel run
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def run_single_fold(
    *,
    fold_id: int,
    outer_train_idx: list[int],
    test_idx: list[int],
    metadata_df,
    labels: list[int],
    early_embedding,
    late_embedding,
    hidden_dims: list[int],
    threshold_grid,
    args,
):
    train_idx, calib_idx = split_inner_train_calibration(
        outer_train_idx,
        labels,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed + fold_id,
    )

    early_train = early_embedding[train_idx]
    early_calib = early_embedding[calib_idx]
    early_test = early_embedding[test_idx]
    y_train = [labels[idx] for idx in train_idx]
    y_calib = [labels[idx] for idx in calib_idx]
    y_test = [labels[idx] for idx in test_idx]

    early_mean, early_std = standardize_fit(early_train)
    early_train_std = apply_standardize(early_train, early_mean, early_std)
    early_calib_std = apply_standardize(early_calib, early_mean, early_std)
    early_test_std = apply_standardize(early_test, early_mean, early_std)

    early_model, early_history = fit_mlp_classifier(
        x_train=early_train_std,
        y_train=y_train,
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed + 1000 + fold_id,
    )

    early_train_probs = predict_positive_prob(
        early_model, early_train_std, device=args.device
    )
    early_calib_probs = predict_positive_prob(
        early_model, early_calib_std, device=args.device
    )
    early_test_probs = predict_positive_prob(
        early_model, early_test_std, device=args.device
    )
    early_sweep = sweep_thresholds(
        y_calib, early_calib_probs, thresholds=threshold_grid
    )
    early_threshold_row = select_stage1_threshold_row(
        early_sweep,
        min_recall=args.stage1_min_recall,
        selection_mode=args.stage1_selection_mode,
    )
    early_threshold = float(early_threshold_row["threshold"])
    early_test_pred = [1 if prob >= early_threshold else 0 for prob in early_test_probs]

    if args.stage2_mode == "ultralowfpr":
        stage2_train_idx = build_ultralowfpr_stage2_train_indices(
            train_idx=train_idx,
            labels=labels,
            stage1_train_probs=early_train_probs,
            stage1_threshold=early_threshold,
            min_hard_negatives=args.min_hard_negatives,
            negative_multiplier=args.stage2_negative_multiplier,
        )
    else:
        stage2_train_idx = build_stage2_train_indices(
            train_idx=train_idx,
            labels=labels,
            stage1_train_probs=early_train_probs,
            stage1_threshold=early_threshold,
            min_hard_negatives=args.min_hard_negatives,
        )

    late_stage2_train = late_embedding[stage2_train_idx]
    late_calib = late_embedding[calib_idx]
    late_test = late_embedding[test_idx]
    late_stage2_labels = [labels[idx] for idx in stage2_train_idx]

    late_mean, late_std = standardize_fit(late_stage2_train)
    late_stage2_train_std = apply_standardize(late_stage2_train, late_mean, late_std)
    late_calib_std = apply_standardize(late_calib, late_mean, late_std)
    late_test_std = apply_standardize(late_test, late_mean, late_std)

    late_model, late_history = fit_mlp_classifier(
        x_train=late_stage2_train_std,
        y_train=late_stage2_labels,
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed + 2000 + fold_id,
    )

    late_calib_probs = predict_positive_prob(
        late_model, late_calib_std, device=args.device
    )
    late_test_probs = predict_positive_prob(
        late_model, late_test_std, device=args.device
    )

    late_sweep = sweep_thresholds(y_calib, late_calib_probs, thresholds=threshold_grid)
    late_threshold_row = select_stage2_threshold_row(
        late_sweep,
        max_fpr=args.stage2_max_fpr,
        selection_mode=args.stage2_selection_mode,
    )
    late_threshold = float(late_threshold_row["threshold"])
    late_test_pred = [1 if prob >= late_threshold else 0 for prob in late_test_probs]

    cascade_sweep = []
    for row in late_sweep:
        preds = apply_cascade(
            early_calib_probs,
            late_calib_probs,
            stage1_threshold=early_threshold,
            stage2_threshold=float(row["threshold"]),
        )
        cascade_sweep.append(
            {
                "threshold": float(row["threshold"]),
                **compute_binary_metrics(y_calib, preds),
            }
        )
    cascade_threshold_row = select_stage2_threshold_row(
        cascade_sweep,
        max_fpr=args.stage2_max_fpr,
        selection_mode=args.stage2_selection_mode,
    )
    cascade_threshold = float(cascade_threshold_row["threshold"])
    cascade_test_pred = apply_cascade(
        early_test_probs,
        late_test_probs,
        stage1_threshold=early_threshold,
        stage2_threshold=cascade_threshold,
    )

    fold_metric_rows = []
    threshold_rows = [
        {
            "fold": fold_id,
            "model": "early",
            "threshold": early_threshold,
            "selection_rule": f"{args.stage1_selection_mode}: recall>={args.stage1_min_recall}",
            "selected_metric": early_threshold_row["precision"],
            "selected_fpr": early_threshold_row["fpr"],
        },
        {
            "fold": fold_id,
            "model": "late",
            "threshold": late_threshold,
            "selection_rule": f"{args.stage2_selection_mode}: fpr<={args.stage2_max_fpr}",
            "selected_metric": late_threshold_row["precision"],
            "selected_fpr": late_threshold_row["fpr"],
        },
        {
            "fold": fold_id,
            "model": "cascade",
            "threshold": cascade_threshold,
            "selection_rule": f"{args.stage2_selection_mode}: fpr<={args.stage2_max_fpr}",
            "selected_metric": cascade_threshold_row["precision"],
            "selected_fpr": cascade_threshold_row["fpr"],
        },
    ]

    for current_model_name, preds in {
        "early": early_test_pred,
        "late": late_test_pred,
        "cascade": cascade_test_pred,
    }.items():
        fold_metric_rows.append(
            {
                "fold": fold_id,
                "model": current_model_name,
                **compute_binary_metrics(y_test, preds),
            }
        )

    fold_df = metadata_df.iloc[test_idx].copy()
    fold_df["fold"] = fold_id
    fold_df["binary_target"] = y_test
    fold_df["early_prob"] = early_test_probs
    fold_df["late_prob"] = late_test_probs
    fold_df["early_pred"] = early_test_pred
    fold_df["late_pred"] = late_test_pred
    fold_df["cascade_pred"] = cascade_test_pred
    fold_df["early_threshold"] = early_threshold
    fold_df["late_threshold"] = late_threshold
    fold_df["cascade_threshold"] = cascade_threshold

    history_rows = [{"fold": fold_id, "model": "early", **row} for row in early_history]
    history_rows.extend(
        {"fold": fold_id, "model": "late", **row} for row in late_history
    )
    return {
        "predictions_df": fold_df,
        "fold_metric_rows": fold_metric_rows,
        "threshold_rows": threshold_rows,
        "history_rows": history_rows,
    }


def write_fold_outputs(fold_output_root: Path, fold_result: dict):
    fold_output_root.mkdir(parents=True, exist_ok=True)
    _sorted_predictions_df(fold_result["predictions_df"]).to_csv(
        fold_output_root / "predictions.csv", index=False
    )
    import pandas as pd

    pd.DataFrame(fold_result["fold_metric_rows"]).to_csv(
        fold_output_root / "fold_metrics.csv", index=False
    )
    pd.DataFrame(fold_result["threshold_rows"]).to_csv(
        fold_output_root / "thresholds.csv", index=False
    )
    pd.DataFrame(fold_result["history_rows"]).to_csv(
        fold_output_root / "training_history.csv", index=False
    )


def aggregate_fold_outputs(
    *,
    output_root: Path,
    model_name: str,
    selection_json: str,
    early_layer: int,
    late_layer: int,
    stage1_min_recall: float,
    stage1_selection_mode: str,
    stage2_max_fpr: float,
    stage2_selection_mode: str,
    stage2_mode: str,
    stage2_negative_multiplier: float,
    embedding_dir: str,
    metadata: str,
    batch_size: int,
    parallel_folds: int,
):
    import pandas as pd

    fold_root = output_root / "fold_runs"
    prediction_frames = []
    fold_metric_rows = []
    threshold_rows = []
    history_rows = []

    for fold_dir in sorted(fold_root.glob("fold_*")):
        prediction_frames.append(pd.read_csv(fold_dir / "predictions.csv"))
        fold_metric_rows.extend(
            pd.read_csv(fold_dir / "fold_metrics.csv").to_dict(orient="records")
        )
        threshold_rows.extend(
            pd.read_csv(fold_dir / "thresholds.csv").to_dict(orient="records")
        )
        history_rows.extend(
            pd.read_csv(fold_dir / "training_history.csv").to_dict(orient="records")
        )

    predictions_df = _sorted_predictions_df(
        pd.concat(prediction_frames, ignore_index=True)
    )
    predictions_df.to_csv(output_root / "predictions.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(output_root / "fold_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_root / "thresholds.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_root / "training_history.csv", index=False)

    summary = build_summary(
        predictions_df=predictions_df,
        fold_metric_rows=fold_metric_rows,
        model_name=model_name,
        early_layer=early_layer,
        late_layer=late_layer,
        stage1_min_recall=stage1_min_recall,
        stage1_selection_mode=stage1_selection_mode,
        stage2_max_fpr=stage2_max_fpr,
        stage2_selection_mode=stage2_selection_mode,
        stage2_mode=stage2_mode,
        stage2_negative_multiplier=stage2_negative_multiplier,
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "selected_layers_used.json").write_text(
        json.dumps(
            {
                "early_layer": early_layer,
                "late_layer": late_layer,
                "selection_json": selection_json,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_readme(
        output_root,
        model_name=model_name,
        embedding_dir=embedding_dir,
        metadata_path=metadata,
        selection_json=selection_json,
        early_layer=early_layer,
        late_layer=late_layer,
        batch_size=batch_size,
        parallel_folds=parallel_folds,
        stage1_min_recall=stage1_min_recall,
        stage1_selection_mode=stage1_selection_mode,
        stage2_max_fpr=stage2_max_fpr,
        stage2_selection_mode=stage2_selection_mode,
    )


def build_subprocess_args(args, fold_id: int) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--embedding-dir",
        args.embedding_dir,
        "--metadata",
        args.metadata,
        "--selection-json",
        args.selection_json,
        "--output-root",
        args.output_root,
        "--model-name",
        args.model_name,
        "--folds",
        str(args.folds),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--calibration-fraction",
        str(args.calibration_fraction),
        "--stage1-min-recall",
        str(args.stage1_min_recall),
        "--stage1-selection-mode",
        args.stage1_selection_mode,
        "--stage2-max-fpr",
        str(args.stage2_max_fpr),
        "--stage2-selection-mode",
        args.stage2_selection_mode,
        "--min-hard-negatives",
        str(args.min_hard_negatives),
        "--stage2-negative-multiplier",
        str(args.stage2_negative_multiplier),
        "--stage2-mode",
        args.stage2_mode,
        "--threshold-grid-size",
        str(args.threshold_grid_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--parallel-folds",
        "1",
        "--fold-index",
        str(fold_id),
    ]
    if args.early_layer is not None:
        cmd.extend(["--early-layer", str(args.early_layer)])
    if args.late_layer is not None:
        cmd.extend(["--late-layer", str(args.late_layer)])
    if args.hidden_dims:
        cmd.extend(["--hidden-dims", *[str(dim) for dim in args.hidden_dims]])
    return cmd


def main():
    import numpy as np
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--early-layer", type=int, default=None)
    parser.add_argument("--late-layer", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--stage1-min-recall", type=float, default=0.99)
    parser.add_argument(
        "--stage1-selection-mode",
        choices=["min_fpr", "max_precision"],
        default="min_fpr",
    )
    parser.add_argument("--stage2-max-fpr", type=float, default=0.0001)
    parser.add_argument(
        "--stage2-selection-mode",
        choices=["max_recall", "max_precision"],
        default="max_recall",
    )
    parser.add_argument("--min-hard-negatives", type=int, default=2048)
    parser.add_argument("--stage2-negative-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--stage2-mode", choices=["standard", "ultralowfpr"], default="standard"
    )
    parser.add_argument("--threshold-grid-size", type=int, default=2001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parallel-folds", type=int, default=1)
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    hidden_dims = normalize_hidden_dims(
        hidden_dims=args.hidden_dims, hidden_dim=args.hidden_dim
    )

    selection_payload = json.loads(
        Path(args.selection_json).read_text(encoding="utf-8")
    )
    early_layer = args.early_layer or int(selection_payload["early_layer"])
    late_layer = args.late_layer or int(selection_payload["late_layer"])

    metadata_df = pd.read_csv(args.metadata)
    labels = build_binary_targets(
        metadata_df["label"].astype(str).tolist(), positive_label="Capsid"
    )
    label_strings = [str(label) for label in labels]
    folds = build_stratified_folds(label_strings, n_splits=args.folds, seed=args.seed)

    early_embedding = np.load(
        Path(args.embedding_dir) / f"layer_{early_layer}.npy", mmap_mode="r"
    )
    late_embedding = np.load(
        Path(args.embedding_dir) / f"layer_{late_layer}.npy", mmap_mode="r"
    )

    threshold_grid = np.linspace(0.0, 1.0, args.threshold_grid_size).tolist()

    if args.aggregate_only:
        aggregate_fold_outputs(
            output_root=output_root,
            model_name=args.model_name,
            selection_json=args.selection_json,
            early_layer=early_layer,
            late_layer=late_layer,
            stage1_min_recall=args.stage1_min_recall,
            stage1_selection_mode=args.stage1_selection_mode,
            stage2_max_fpr=args.stage2_max_fpr,
            stage2_selection_mode=args.stage2_selection_mode,
            stage2_mode=args.stage2_mode,
            stage2_negative_multiplier=args.stage2_negative_multiplier,
            embedding_dir=args.embedding_dir,
            metadata=args.metadata,
            batch_size=args.batch_size,
            parallel_folds=max(1, args.parallel_folds),
        )
        return

    if args.parallel_folds > 1 and args.fold_index is None:
        processes = []
        for fold_id in range(1, min(args.parallel_folds, len(folds)) + 1):
            processes.append(
                subprocess.Popen(build_subprocess_args(args, fold_id=fold_id))
            )
        failures = []
        for fold_id, process in enumerate(processes, start=1):
            rc = process.wait()
            if rc != 0:
                failures.append((fold_id, rc))
        if failures:
            raise RuntimeError(f"parallel fold workers failed: {failures}")
        aggregate_fold_outputs(
            output_root=output_root,
            model_name=args.model_name,
            selection_json=args.selection_json,
            early_layer=early_layer,
            late_layer=late_layer,
            stage1_min_recall=args.stage1_min_recall,
            stage1_selection_mode=args.stage1_selection_mode,
            stage2_max_fpr=args.stage2_max_fpr,
            stage2_selection_mode=args.stage2_selection_mode,
            stage2_mode=args.stage2_mode,
            stage2_negative_multiplier=args.stage2_negative_multiplier,
            embedding_dir=args.embedding_dir,
            metadata=args.metadata,
            batch_size=args.batch_size,
            parallel_folds=args.parallel_folds,
        )
        return

    if args.fold_index is not None:
        if args.fold_index < 1 or args.fold_index > len(folds):
            raise ValueError(f"fold_index must be between 1 and {len(folds)}")
        outer_train_idx, test_idx = folds[args.fold_index - 1]
        fold_result = run_single_fold(
            fold_id=args.fold_index,
            outer_train_idx=outer_train_idx,
            test_idx=test_idx,
            metadata_df=metadata_df,
            labels=labels,
            early_embedding=early_embedding,
            late_embedding=late_embedding,
            hidden_dims=hidden_dims,
            threshold_grid=threshold_grid,
            args=args,
        )
        write_fold_outputs(
            output_root / "fold_runs" / f"fold_{args.fold_index}", fold_result
        )
        return

    prediction_frames = []
    fold_metric_rows = []
    threshold_rows = []
    history_rows = []
    for fold_id, (outer_train_idx, test_idx) in enumerate(folds, start=1):
        fold_result = run_single_fold(
            fold_id=fold_id,
            outer_train_idx=outer_train_idx,
            test_idx=test_idx,
            metadata_df=metadata_df,
            labels=labels,
            early_embedding=early_embedding,
            late_embedding=late_embedding,
            hidden_dims=hidden_dims,
            threshold_grid=threshold_grid,
            args=args,
        )
        prediction_frames.append(fold_result["predictions_df"])
        fold_metric_rows.extend(fold_result["fold_metric_rows"])
        threshold_rows.extend(fold_result["threshold_rows"])
        history_rows.extend(fold_result["history_rows"])

    predictions_df = _sorted_predictions_df(
        pd.concat(prediction_frames, ignore_index=True)
    )
    predictions_df.to_csv(output_root / "predictions.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(output_root / "fold_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_root / "thresholds.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_root / "training_history.csv", index=False)

    summary = build_summary(
        predictions_df=predictions_df,
        fold_metric_rows=fold_metric_rows,
        model_name=args.model_name,
        early_layer=early_layer,
        late_layer=late_layer,
        stage1_min_recall=args.stage1_min_recall,
        stage1_selection_mode=args.stage1_selection_mode,
        stage2_max_fpr=args.stage2_max_fpr,
        stage2_selection_mode=args.stage2_selection_mode,
        stage2_mode=args.stage2_mode,
        stage2_negative_multiplier=args.stage2_negative_multiplier,
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "selected_layers_used.json").write_text(
        json.dumps(
            {
                "early_layer": early_layer,
                "late_layer": late_layer,
                "selection_json": args.selection_json,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_readme(
        output_root,
        model_name=args.model_name,
        embedding_dir=args.embedding_dir,
        metadata_path=args.metadata,
        selection_json=args.selection_json,
        early_layer=early_layer,
        late_layer=late_layer,
        batch_size=args.batch_size,
        parallel_folds=max(1, args.parallel_folds),
        stage1_min_recall=args.stage1_min_recall,
        stage1_selection_mode=args.stage1_selection_mode,
        stage2_max_fpr=args.stage2_max_fpr,
        stage2_selection_mode=args.stage2_selection_mode,
    )


if __name__ == "__main__":
    main()
