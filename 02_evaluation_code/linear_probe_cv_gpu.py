from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path


def compute_metrics_from_pairs(y_true, y_pred, labels):
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    size = len(labels)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[label_to_idx[true_label]][label_to_idx[pred_label]] += 1

    total = len(y_true)
    correct = sum(confusion[i][i] for i in range(size))
    accuracy = correct / total if total else 0.0

    per_class_f1 = []
    supports = []
    for idx in range(size):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(size) if row != idx)
        fn = sum(confusion[idx][col] for col in range(size) if col != idx)
        support = sum(confusion[idx])
        supports.append(support)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class_f1.append(f1)

    macro_f1 = sum(per_class_f1) / size if size else 0.0
    weighted_f1 = (
        sum(f1 * support for f1, support in zip(per_class_f1, supports)) / total
        if total
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }, confusion


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def standardize_fit(x_train):
    import numpy as np

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardize(x, mean, std):
    return ((x - mean) / std).astype("float32", copy=False)


def encode_labels(labels):
    unique = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique)}
    encoded = [label_to_idx[label] for label in labels]
    return encoded, label_to_idx


def build_stratified_folds(labels, n_splits=5, seed=42):
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    fold_tests = [[] for _ in range(n_splits)]
    for indices in label_to_indices.values():
        if len(indices) < n_splits:
            raise ValueError("each class must have at least n_splits samples")
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for position, idx in enumerate(shuffled):
            fold_tests[position % n_splits].append(idx)

    all_indices = set(range(len(labels)))
    return [
        (sorted(all_indices.difference(test_idx)), sorted(test_idx))
        for test_idx in fold_tests
    ]


def train_one_fold(
    x_train, y_train, x_test, y_test, epochs, lr, weight_decay, batch_size, device, seed
):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_classes = int(max(y_train) + 1)
    feature_dim = int(x_train.shape[1])
    model = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

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

            batch_size_now = int(batch_y.shape[0])
            epoch_loss += float(loss.item()) * batch_size_now
            seen += batch_size_now

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(seen, 1),
            }
        )

    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(x_test.astype(np.float32, copy=False)).to(device)
        logits = model(test_tensor)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()

    return preds, history


def main():
    import numpy as np
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--folds", type=int, default=5)
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

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    df = pd.read_csv(args.metadata)
    if "label" not in df.columns:
        raise ValueError("metadata must contain label column")
    if "prot_id" not in df.columns:
        raise ValueError("metadata must contain prot_id column")

    x = np.load(args.embedding)
    if len(df) != int(x.shape[0]):
        raise ValueError(
            f"metadata rows ({len(df)}) do not match embedding rows ({x.shape[0]})"
        )

    labels = df["label"].astype(str).tolist()
    encoded_labels, label_to_idx = encode_labels(labels)
    folds = build_stratified_folds(labels, n_splits=args.folds, seed=args.seed)

    all_predictions = []
    fold_metrics = []
    histories = []

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = [encoded_labels[i] for i in train_idx]
        y_test = [encoded_labels[i] for i in test_idx]

        mean, std = standardize_fit(x_train)
        x_train = apply_standardize(x_train, mean, std)
        x_test = apply_standardize(x_test, mean, std)

        preds, history = train_one_fold(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed + fold_id,
        )
        histories.extend(
            {
                "fold": fold_id,
                **row,
            }
            for row in history
        )

        metrics, confusion = compute_metrics_from_pairs(
            y_true=y_test,
            y_pred=preds,
            labels=list(range(len(label_to_idx))),
        )
        fold_metrics.append(
            {
                "fold": fold_id,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **metrics,
            }
        )

        inverse = {value: key for key, value in label_to_idx.items()}
        fold_df = df.iloc[test_idx].copy()
        fold_df["fold"] = fold_id
        fold_df["true_label"] = [inverse[y] for y in y_test]
        fold_df["pred_label"] = [inverse[y] for y in preds]
        fold_df["true_label_id"] = y_test
        fold_df["pred_label_id"] = preds
        all_predictions.append(fold_df)

        cm_df = pd.DataFrame(
            confusion,
            index=[inverse[idx] for idx in range(len(inverse))],
            columns=[inverse[idx] for idx in range(len(inverse))],
        )
        cm_df.to_csv(output_dir / f"confusion_matrix_fold{fold_id}.csv")

    metric_names = ["accuracy", "macro_f1", "weighted_f1"]
    summary = {}
    for metric_name in metric_names:
        values = [row[metric_name] for row in fold_metrics]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[f"{metric_name}_mean"] = mean_value
        summary[f"{metric_name}_std"] = math.sqrt(variance)

    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.concat(all_predictions, axis=0).to_csv(
        output_dir / "predictions.csv", index=False
    )
    pd.DataFrame(histories).to_csv(output_dir / "training_history.csv", index=False)

    run_config = {
        "embedding": args.embedding,
        "metadata": args.metadata,
        "output": args.output,
        "model_name": args.model_name,
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
        "n_samples": len(df),
        "feature_dim": int(x.shape[1]),
        "label_to_idx": label_to_idx,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps({"summary": summary, "output": str(output_dir)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
