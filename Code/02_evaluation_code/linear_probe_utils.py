from __future__ import annotations

import math
import random
from typing import Iterable


LAYER_KEYS = (
    "num_hidden_layers",
    "n_layer",
    "num_layers",
    "encoder_layers",
    "num_hidden_blocks",
    "depth",
)


def resolve_penultimate_layer(config_like: dict) -> int:
    for key in LAYER_KEYS:
        value = config_like.get(key)
        if value is None:
            continue
        total_layers = int(value)
        if total_layers < 2:
            raise ValueError("model must have at least 2 layers")
        return total_layers - 2
    raise ValueError("could not determine layer count")


def build_stratified_folds(labels: Iterable[str], n_splits: int = 5, seed: int = 42):
    labels = list(labels)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(labels) < n_splits:
        raise ValueError("not enough samples for requested folds")

    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    fold_tests = [[] for _ in range(n_splits)]
    for indices in label_to_indices.values():
        if len(indices) < n_splits:
            raise ValueError("each class must have at least n_splits samples")
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for position, idx in enumerate(shuffled):
            fold_tests[position % n_splits].append(idx)

    all_indices = set(range(len(labels)))
    folds = []
    for test_idx in fold_tests:
        test_idx = sorted(test_idx)
        train_idx = sorted(all_indices.difference(test_idx))
        folds.append((train_idx, test_idx))
    return folds


def summarize_metrics(rows: list[dict], metric_names: Iterable[str]):
    summary = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[f"{metric_name}_mean"] = mean
        summary[f"{metric_name}_std"] = math.sqrt(variance)
    return summary
