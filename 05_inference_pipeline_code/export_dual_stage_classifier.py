from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

try:
    from experiment_tools.layer_selection_utils import (
        build_binary_targets,
        select_best_row_by_max_fpr,
        select_best_row_by_max_fpr_max_precision,
        select_best_row_by_min_recall,
        select_best_row_by_min_recall_max_precision,
        sweep_thresholds,
    )
    from experiment_tools.linear_probe_cv_gpu import apply_standardize, standardize_fit
    from experiment_tools.train_dual_stage_low_fpr import (
        build_stage2_train_indices,
        build_ultralowfpr_stage2_train_indices,
        fit_mlp_classifier,
        normalize_hidden_dims,
        predict_positive_prob,
    )
except ImportError:  # pragma: no cover
    from layer_selection_utils import (
        build_binary_targets,
        select_best_row_by_max_fpr,
        select_best_row_by_max_fpr_max_precision,
        select_best_row_by_min_recall,
        select_best_row_by_min_recall_max_precision,
        sweep_thresholds,
    )
    from linear_probe_cv_gpu import apply_standardize, standardize_fit
    from train_dual_stage_low_fpr import (
        build_stage2_train_indices,
        build_ultralowfpr_stage2_train_indices,
        fit_mlp_classifier,
        normalize_hidden_dims,
        predict_positive_prob,
    )


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_export_config(
    model_name: str,
    early_layer: int,
    late_layer: int,
    stage1_threshold: float,
    stage2_threshold: float,
    selection_json: str,
    metadata_path: str,
    hidden_dims: list[int],
    stage1_selection_mode: str,
    stage2_selection_mode: str,
):
    return {
        "model_name": model_name,
        "early_layer": int(early_layer),
        "late_layer": int(late_layer),
        "stage1_threshold": float(stage1_threshold),
        "stage2_threshold": float(stage2_threshold),
        "selection_json": selection_json,
        "metadata_path": metadata_path,
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "stage1_selection_mode": stage1_selection_mode,
        "stage2_selection_mode": stage2_selection_mode,
    }


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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--stage1-min-recall", type=float, default=0.99)
    parser.add_argument(
        "--stage1-selection-mode",
        choices=["min_fpr", "max_precision"],
        default="min_fpr",
    )
    parser.add_argument("--stage2-max-fpr", type=float, default=0.000025)
    parser.add_argument(
        "--stage2-selection-mode",
        choices=["max_recall", "max_precision"],
        default="max_recall",
    )
    parser.add_argument("--min-hard-negatives", type=int, default=20000)
    parser.add_argument("--stage2-negative-multiplier", type=float, default=8.0)
    parser.add_argument(
        "--stage2-mode", choices=["standard", "ultralowfpr"], default="ultralowfpr"
    )
    parser.add_argument("--threshold-grid-size", type=int, default=10001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    hidden_dims = normalize_hidden_dims(
        hidden_dims=args.hidden_dims, hidden_dim=args.hidden_dim
    )

    selection_payload = json.loads(
        Path(args.selection_json).read_text(encoding="utf-8")
    )
    early_layer = int(selection_payload["early_layer"])
    late_layer = int(selection_payload["late_layer"])

    metadata_df = pd.read_csv(args.metadata)
    labels = build_binary_targets(
        metadata_df["label"].astype(str).tolist(), positive_label="Capsid"
    )
    train_idx = list(range(len(labels)))

    early_embedding = np.load(
        Path(args.embedding_dir) / f"layer_{early_layer}.npy", mmap_mode="r"
    )
    late_embedding = np.load(
        Path(args.embedding_dir) / f"layer_{late_layer}.npy", mmap_mode="r"
    )
    threshold_grid = np.linspace(0.0, 1.0, args.threshold_grid_size).tolist()

    early_x = early_embedding[train_idx]
    early_y = [labels[idx] for idx in train_idx]
    early_mean, early_std = standardize_fit(early_x)
    early_x_std = apply_standardize(early_x, early_mean, early_std)
    early_model, early_history = fit_mlp_classifier(
        x_train=early_x_std,
        y_train=early_y,
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed + 1000,
    )
    early_probs = predict_positive_prob(early_model, early_x_std, device=args.device)
    early_sweep = sweep_thresholds(labels, early_probs, thresholds=threshold_grid)
    if args.stage1_selection_mode == "max_precision":
        early_threshold_row = select_best_row_by_min_recall_max_precision(
            early_sweep,
            min_recall=args.stage1_min_recall,
        )
    else:
        early_threshold_row = select_best_row_by_min_recall(
            early_sweep, min_recall=args.stage1_min_recall
        )
    early_threshold = float(early_threshold_row["threshold"])

    if args.stage2_mode == "ultralowfpr":
        stage2_train_idx = build_ultralowfpr_stage2_train_indices(
            train_idx=train_idx,
            labels=labels,
            stage1_train_probs=early_probs,
            stage1_threshold=early_threshold,
            min_hard_negatives=args.min_hard_negatives,
            negative_multiplier=args.stage2_negative_multiplier,
        )
    else:
        stage2_train_idx = build_stage2_train_indices(
            train_idx=train_idx,
            labels=labels,
            stage1_train_probs=early_probs,
            stage1_threshold=early_threshold,
            min_hard_negatives=args.min_hard_negatives,
        )

    late_x = late_embedding[stage2_train_idx]
    late_y = [labels[idx] for idx in stage2_train_idx]
    late_mean, late_std = standardize_fit(late_x)
    late_x_std = apply_standardize(late_x, late_mean, late_std)
    late_model, late_history = fit_mlp_classifier(
        x_train=late_x_std,
        y_train=late_y,
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed + 2000,
    )

    all_late_x = apply_standardize(late_embedding[train_idx], late_mean, late_std)
    late_probs = predict_positive_prob(late_model, all_late_x, device=args.device)
    late_sweep = sweep_thresholds(labels, late_probs, thresholds=threshold_grid)
    if args.stage2_selection_mode == "max_precision":
        late_threshold_row = select_best_row_by_max_fpr_max_precision(
            late_sweep, max_fpr=args.stage2_max_fpr
        )
    else:
        late_threshold_row = select_best_row_by_max_fpr(
            late_sweep, max_fpr=args.stage2_max_fpr
        )
    late_threshold = float(late_threshold_row["threshold"])

    torch.save(
        {
            "state_dict": early_model.state_dict(),
            "hidden_dim": hidden_dims[0],
            "hidden_dims": hidden_dims,
            "input_dim": int(early_x.shape[1]),
        },
        models_dir / "early_head.pt",
    )
    torch.save(
        {
            "state_dict": late_model.state_dict(),
            "hidden_dim": hidden_dims[0],
            "hidden_dims": hidden_dims,
            "input_dim": int(late_x.shape[1]),
        },
        models_dir / "late_head.pt",
    )
    np.save(models_dir / "early_mean.npy", early_mean)
    np.save(models_dir / "early_std.npy", early_std)
    np.save(models_dir / "late_mean.npy", late_mean)
    np.save(models_dir / "late_std.npy", late_std)

    export_config = build_export_config(
        model_name=args.model_name,
        early_layer=early_layer,
        late_layer=late_layer,
        stage1_threshold=early_threshold,
        stage2_threshold=late_threshold,
        selection_json=args.selection_json,
        metadata_path=args.metadata,
        hidden_dims=hidden_dims,
        stage1_selection_mode=args.stage1_selection_mode,
        stage2_selection_mode=args.stage2_selection_mode,
    )
    export_config.update(
        {
            "stage1_min_recall": args.stage1_min_recall,
            "stage1_selection_mode": args.stage1_selection_mode,
            "stage2_max_fpr": args.stage2_max_fpr,
            "stage2_selection_mode": args.stage2_selection_mode,
            "stage2_mode": args.stage2_mode,
            "stage2_negative_multiplier": args.stage2_negative_multiplier,
            "min_hard_negatives": args.min_hard_negatives,
            "hidden_dim": hidden_dims[0],
            "hidden_dims": hidden_dims,
            "threshold_grid_size": args.threshold_grid_size,
        }
    )
    (output_root / "export_config.json").write_text(
        json.dumps(export_config, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"model": "early", **row} for row in early_history]
        + [{"model": "late", **row} for row in late_history]
    ).to_csv(output_root / "training_history_final.csv", index=False)


if __name__ == "__main__":
    main()
