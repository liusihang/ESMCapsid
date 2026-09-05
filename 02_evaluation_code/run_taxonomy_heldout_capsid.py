from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

try:
    from experiment_tools.linear_probe_cv_gpu import apply_standardize, standardize_fit
    from experiment_tools.taxonomy_heldout_utils import (
        build_leave_one_taxon_out_splits,
        filter_groups_by_min_size,
    )
except ImportError:  # pragma: no cover - remote flat-script fallback
    from linear_probe_cv_gpu import apply_standardize, standardize_fit
    from taxonomy_heldout_utils import (
        build_leave_one_taxon_out_splits,
        filter_groups_by_min_size,
    )


MODEL_RENAMES = {"ESMCapsid-S": "ESMCapsid-S"}
LEVELS = ["Realm", "Kingdom", "Phylum", "Class"]


def normalize_model_name(model_name: str) -> str:
    return MODEL_RENAMES.get(model_name, model_name)


def taxonomy_result_dir(results_root: str, model_name: str) -> str:
    return str(Path(results_root) / model_name)


def _mean_std(values):
    mean_value = sum(values) / len(values) if values else 0.0
    variance = (
        sum((value - mean_value) ** 2 for value in values) / len(values)
        if values
        else 0.0
    )
    return mean_value, math.sqrt(variance)


def summarize_level_rows(level: str, rows: list[dict], model_name: str = ""):
    capsid_f1_mean, capsid_f1_std = _mean_std(
        [row.get("capsid_f1", 0.0) for row in rows]
    )
    balanced_accuracy_mean, balanced_accuracy_std = _mean_std(
        [row.get("balanced_accuracy", 0.0) for row in rows]
    )
    capsid_precision_mean, capsid_precision_std = _mean_std(
        [row.get("capsid_precision", 0.0) for row in rows]
    )
    capsid_recall_mean, capsid_recall_std = _mean_std(
        [row.get("capsid_recall", 0.0) for row in rows]
    )
    accuracy_mean, accuracy_std = _mean_std([row.get("accuracy", 0.0) for row in rows])
    return {
        "model": model_name,
        "level": level,
        "heldout_taxa": len(rows),
        "capsid_f1_mean": capsid_f1_mean,
        "capsid_f1_std": capsid_f1_std,
        "balanced_accuracy_mean": balanced_accuracy_mean,
        "balanced_accuracy_std": balanced_accuracy_std,
        "capsid_precision_mean": capsid_precision_mean,
        "capsid_precision_std": capsid_precision_std,
        "capsid_recall_mean": capsid_recall_mean,
        "capsid_recall_std": capsid_recall_std,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
    }


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stratified_train_test_split(
    rows: list[dict], label_key: str, test_fraction: float, seed: int
):
    by_label = {}
    for row in rows:
        by_label.setdefault(row[label_key], []).append(dict(row))
    rng = random.Random(seed)
    train_rows = []
    test_rows = []
    for label in sorted(by_label):
        chunk = by_label[label]
        rng.shuffle(chunk)
        test_n = max(1, int(round(len(chunk) * test_fraction)))
        test_rows.extend(chunk[:test_n])
        train_rows.extend(chunk[test_n:])
    return train_rows, test_rows


def compute_binary_metrics(y_true: list[int], y_pred: list[int]):
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "capsid_precision": precision,
        "capsid_recall": recall,
        "specificity_non_capsid": specificity,
        "capsid_f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def _load_inputs(embedding_path: str, metadata_path: str, taxonomy_path: str):
    import numpy as np
    import pandas as pd

    x = np.load(embedding_path)
    metadata_df = pd.read_csv(metadata_path)
    taxonomy_df = pd.read_csv(taxonomy_path)
    if len(metadata_df) != int(x.shape[0]):
        raise ValueError(
            f"metadata rows ({len(metadata_df)}) do not match embedding rows ({x.shape[0]})"
        )
    return x, metadata_df, taxonomy_df


def _make_embedding_rows(x, metadata_df):
    rows = []
    for idx, row in metadata_df.iterrows():
        rows.append(
            {
                "row_index": int(idx),
                "prot_id": str(row["prot_id"]),
                "label": str(row["label"]),
            }
        )
    return rows


def _build_taxonomy_rows(taxonomy_df, level: str):
    if level not in taxonomy_df.columns:
        raise ValueError(f"taxonomy file missing column {level}")

    rows = []
    grouped = (
        taxonomy_df[["prot_id", level]]
        .dropna(subset=["prot_id", level])
        .assign(prot_id=lambda df: df["prot_id"].astype(str))
        .groupby("prot_id")[level]
    )
    for prot_id, values in grouped:
        unique_values = sorted(
            {str(value).strip() for value in values if str(value).strip()}
        )
        if len(unique_values) != 1:
            continue
        rows.append({"prot_id": prot_id, level: unique_values[0]})
    return rows


def _prepare_level_splits(embedding_rows, taxonomy_df, level: str, min_group_size: int):
    capsid_ids = {row["prot_id"] for row in embedding_rows if row["label"] == "Capsid"}
    taxonomy_rows = _build_taxonomy_rows(taxonomy_df, level)
    valid_rows = filter_groups_by_min_size(
        taxonomy_rows,
        level=level,
        min_size=min_group_size,
        allowed_ids=capsid_ids,
    )
    return build_leave_one_taxon_out_splits(valid_rows, level=level)


def _build_negative_split(embedding_rows, test_fraction: float, seed: int):
    negative_rows = [row for row in embedding_rows if row["label"] != "Capsid"]
    return stratified_train_test_split(
        negative_rows,
        label_key="label",
        test_fraction=test_fraction,
        seed=seed,
    )


def train_binary_probe(
    x_train,
    y_train,
    x_test,
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

    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32, copy=False)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
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
        test_tensor = torch.from_numpy(x_test.astype(np.float32, copy=False)).to(device)
        logits = model(test_tensor)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()

    return preds, history


def run_level(
    level: str,
    embedding_rows,
    x,
    taxonomy_df,
    negative_train_rows,
    negative_test_rows,
    args,
):
    import pandas as pd

    prot_to_index = {row["prot_id"]: row["row_index"] for row in embedding_rows}
    splits = _prepare_level_splits(
        embedding_rows=embedding_rows,
        taxonomy_df=taxonomy_df,
        level=level,
        min_group_size=args.min_group_size,
    )

    metric_rows = []
    prediction_frames = []
    history_rows = []

    negative_train_ids = [row["prot_id"] for row in negative_train_rows]
    negative_test_ids = [row["prot_id"] for row in negative_test_rows]
    negative_test_size = len(negative_test_ids)

    for split_id, split in enumerate(splits, start=1):
        train_ids = split["train_positive_ids"] + negative_train_ids
        test_ids = split["test_positive_ids"] + negative_test_ids

        train_idx = [prot_to_index[prot_id] for prot_id in train_ids]
        test_idx = [prot_to_index[prot_id] for prot_id in test_ids]
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = [1] * len(split["train_positive_ids"]) + [0] * len(negative_train_ids)
        y_test = [1] * len(split["test_positive_ids"]) + [0] * len(negative_test_ids)

        mean, std = standardize_fit(x_train)
        x_train = apply_standardize(x_train, mean, std)
        x_test = apply_standardize(x_test, mean, std)

        preds, history = train_binary_probe(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed + split_id,
        )
        history_rows.extend(
            {
                "level": level,
                "heldout_taxon": split["heldout_taxon"],
                "split_id": split_id,
                **row,
            }
            for row in history
        )

        metrics = compute_binary_metrics(y_test, preds)
        metric_rows.append(
            {
                "model": args.model_name,
                "level": level,
                "heldout_taxon": split["heldout_taxon"],
                "split_id": split_id,
                "n_train_positive": len(split["train_positive_ids"]),
                "n_test_positive": len(split["test_positive_ids"]),
                "n_train_negative": len(negative_train_ids),
                "n_test_negative": negative_test_size,
                **metrics,
            }
        )

        pred_df = pd.DataFrame(
            {
                "model": args.model_name,
                "level": level,
                "heldout_taxon": split["heldout_taxon"],
                "prot_id": test_ids,
                "true_binary": y_test,
                "pred_binary": preds,
                "true_label": [
                    "Capsid" if value == 1 else "NonCapsid" for value in y_test
                ],
                "pred_label": [
                    "Capsid" if value == 1 else "NonCapsid" for value in preds
                ],
            }
        )
        prediction_frames.append(pred_df)

    summary_row = summarize_level_rows(level, metric_rows, model_name=args.model_name)
    summary_row["eligible_taxa"] = len(splits)
    return summary_row, metric_rows, prediction_frames, history_rows


def main():
    import numpy as np
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--levels", default="Realm,Kingdom,Phylum,Class")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--negative-test-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    x, metadata_df, taxonomy_df = _load_inputs(
        args.embedding, args.metadata, args.taxonomy
    )
    embedding_rows = _make_embedding_rows(x, metadata_df)
    negative_train_rows, negative_test_rows = _build_negative_split(
        embedding_rows,
        test_fraction=args.negative_test_frac,
        seed=args.seed,
    )

    level_rows = []
    all_metric_rows = []
    all_prediction_frames = []
    all_history_rows = []

    levels = [level.strip() for level in args.levels.split(",") if level.strip()]
    for level in levels:
        summary_row, metric_rows, prediction_frames, history_rows = run_level(
            level=level,
            embedding_rows=embedding_rows,
            x=x,
            taxonomy_df=taxonomy_df,
            negative_train_rows=negative_train_rows,
            negative_test_rows=negative_test_rows,
            args=args,
        )
        level_rows.append(summary_row)
        all_metric_rows.extend(metric_rows)
        all_prediction_frames.extend(prediction_frames)
        all_history_rows.extend(history_rows)

    pd.DataFrame(level_rows).to_csv(output_dir / "level_summary.csv", index=False)
    pd.DataFrame(all_metric_rows).to_csv(
        output_dir / "per_taxon_metrics.csv", index=False
    )
    pd.concat(all_prediction_frames, axis=0).to_csv(
        output_dir / "predictions.csv", index=False
    )
    pd.DataFrame(all_history_rows).to_csv(
        output_dir / "training_history.csv", index=False
    )

    run_config = {
        "embedding": args.embedding,
        "metadata": args.metadata,
        "taxonomy": args.taxonomy,
        "output": args.output,
        "model_name": args.model_name,
        "display_name": normalize_model_name(args.model_name),
        "levels": levels,
        "min_group_size": args.min_group_size,
        "negative_test_frac": args.negative_test_frac,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "n_embedding_rows": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "negative_train_rows": len(negative_train_rows),
        "negative_test_rows": len(negative_test_rows),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "level_summary.json").write_text(
        json.dumps(level_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {"output": str(output_dir), "levels": level_rows}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
