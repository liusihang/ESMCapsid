from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

try:
    from experiment_tools.layer_selection_utils import (
        build_binary_targets,
        compute_binary_metrics,
        mean_std,
        select_best_row_by_max_fpr,
        select_best_row_by_min_recall,
        sweep_thresholds,
    )
    from experiment_tools.linear_probe_cv_gpu import apply_standardize, standardize_fit
    from experiment_tools.linear_probe_utils import build_stratified_folds
except ImportError:  # pragma: no cover - remote flat-script fallback
    from layer_selection_utils import (
        build_binary_targets,
        compute_binary_metrics,
        mean_std,
        select_best_row_by_max_fpr,
        select_best_row_by_min_recall,
        sweep_thresholds,
    )
    from linear_probe_cv_gpu import apply_standardize, standardize_fit
    from linear_probe_utils import build_stratified_folds


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_binary_probe(
    x_train,
    y_train,
    x_eval,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    seed: int,
):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    feature_dim = int(x_train.shape[1])
    model = nn.Linear(feature_dim, 2).to(device)

    negatives = max(sum(1 for label in y_train if label == 0), 1)
    positives = max(sum(1 for label in y_train if label == 1), 1)
    class_weights = torch.tensor(
        [len(y_train) / (2.0 * negatives), len(y_train) / (2.0 * positives)],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32, copy=False)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
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

    model.eval()
    with torch.no_grad():
        eval_tensor = torch.from_numpy(x_eval.astype(np.float32, copy=False)).to(device)
        logits = model(eval_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    return probs.tolist(), history


def _rate_key(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "_")


def build_summary(
    y_true: list[int],
    y_score: list[float],
    sweep_rows: list[dict[str, float]],
    layer: int,
) -> dict:
    default_row = next(
        (row for row in sweep_rows if abs(float(row["threshold"]) - 0.5) < 1e-9),
        None,
    )
    if default_row is None:
        default_row = min(
            sweep_rows, key=lambda row: abs(float(row["threshold"]) - 0.5)
        )

    summary = {
        "layer": int(layer),
        "n_samples": len(y_true),
        "n_positive": int(sum(y_true)),
        "n_negative": int(len(y_true) - sum(y_true)),
        "default_threshold": float(default_row["threshold"]),
        "accuracy": float(default_row["accuracy"]),
        "precision": float(default_row["precision"]),
        "recall": float(default_row["recall"]),
        "specificity": float(default_row["specificity"]),
        "fpr": float(default_row["fpr"]),
        "f1": float(default_row["f1"]),
        "balanced_accuracy": float(default_row["balanced_accuracy"]),
    }

    for fpr_cap in (1e-2, 1e-3, 1e-4):
        row = select_best_row_by_max_fpr(sweep_rows, max_fpr=fpr_cap)
        suffix = _rate_key(fpr_cap)
        summary[f"threshold_at_fpr_{suffix}"] = float(row["threshold"])
        summary[f"precision_at_fpr_{suffix}"] = float(row["precision"])
        summary[f"recall_at_fpr_{suffix}"] = float(row["recall"])
        summary[f"fpr_at_fpr_{suffix}"] = float(row["fpr"])

    for recall_floor in (0.99, 0.95):
        row = select_best_row_by_min_recall(sweep_rows, min_recall=recall_floor)
        suffix = _rate_key(recall_floor)
        summary[f"threshold_at_recall_{suffix}"] = float(row["threshold"])
        summary[f"precision_at_recall_{suffix}"] = float(row["precision"])
        summary[f"recall_at_recall_{suffix}"] = float(row["recall"])
        summary[f"fpr_at_recall_{suffix}"] = float(row["fpr"])

    return summary


def main():
    import numpy as np
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--positive-label", default="Capsid")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threshold-grid-size", type=int, default=2001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    metadata_df = pd.read_csv(args.metadata)
    binary_targets = build_binary_targets(
        metadata_df["label"].astype(str).tolist(), positive_label=args.positive_label
    )
    label_strings = [str(target) for target in binary_targets]
    folds = build_stratified_folds(label_strings, n_splits=args.folds, seed=args.seed)

    layer_summaries = []
    for layer_text in [part.strip() for part in args.layers.split(",") if part.strip()]:
        layer = int(layer_text)
        embedding_path = Path(args.embedding_dir) / f"layer_{layer}.npy"
        x = np.load(embedding_path)
        if len(metadata_df) != int(x.shape[0]):
            raise ValueError(
                f"metadata rows ({len(metadata_df)}) do not match embedding rows ({x.shape[0]}) for layer {layer}"
            )

        layer_dir = output_root / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        prediction_frames = []
        fold_metric_rows = []
        history_rows = []
        pooled_true: list[int] = []
        pooled_score: list[float] = []

        for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
            x_train = x[train_idx]
            x_test = x[test_idx]
            y_train = [binary_targets[idx] for idx in train_idx]
            y_test = [binary_targets[idx] for idx in test_idx]

            mean, std = standardize_fit(x_train)
            x_train = apply_standardize(x_train, mean, std)
            x_test = apply_standardize(x_test, mean, std)

            probs, history = train_binary_probe(
                x_train=x_train,
                y_train=y_train,
                x_eval=x_test,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed + fold_id + layer,
            )

            preds = [1 if prob >= 0.5 else 0 for prob in probs]
            metrics = compute_binary_metrics(y_test, preds)
            fold_metric_rows.append({"fold": fold_id, **metrics})
            history_rows.extend({"fold": fold_id, **row} for row in history)

            fold_df = metadata_df.iloc[test_idx].copy()
            fold_df["fold"] = fold_id
            fold_df["binary_target"] = y_test
            fold_df["positive_prob"] = probs
            fold_df["pred_binary"] = preds
            prediction_frames.append(fold_df)

            pooled_true.extend(y_test)
            pooled_score.extend(probs)

        predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(
            ["fold", "prot_id"]
        )
        predictions_df.to_csv(layer_dir / "predictions.csv", index=False)
        pd.DataFrame(fold_metric_rows).to_csv(
            layer_dir / "fold_metrics.csv", index=False
        )
        pd.DataFrame(history_rows).to_csv(
            layer_dir / "training_history.csv", index=False
        )

        thresholds = np.linspace(0.0, 1.0, args.threshold_grid_size)
        sweep_rows = sweep_thresholds(
            pooled_true, pooled_score, thresholds=thresholds.tolist()
        )
        pd.DataFrame(sweep_rows).to_csv(layer_dir / "threshold_sweep.csv", index=False)

        summary = build_summary(pooled_true, pooled_score, sweep_rows, layer=layer)
        accuracy_mean, accuracy_std = mean_std(
            [row["accuracy"] for row in fold_metric_rows]
        )
        precision_mean, precision_std = mean_std(
            [row["precision"] for row in fold_metric_rows]
        )
        recall_mean, recall_std = mean_std([row["recall"] for row in fold_metric_rows])
        fpr_mean, fpr_std = mean_std([row["fpr"] for row in fold_metric_rows])
        summary.update(
            {
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "precision_mean": precision_mean,
                "precision_std": precision_std,
                "recall_mean": recall_mean,
                "recall_std": recall_std,
                "fpr_mean": fpr_mean,
                "fpr_std": fpr_std,
            }
        )
        (layer_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        layer_summaries.append(summary)

    if layer_summaries:
        pd.DataFrame(layer_summaries).sort_values("layer").to_csv(
            output_root / "layer_summaries_partial.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
