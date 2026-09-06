from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def build_binary_targets(
    labels: Iterable[str], positive_label: str = "Capsid"
) -> list[int]:
    return [1 if str(label) == positive_label else 0 for label in labels]


def compute_binary_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
    }


def evaluate_threshold(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> dict[str, float]:
    y_pred = [1 if float(score) >= threshold else 0 for score in y_score]
    metrics = compute_binary_metrics(y_true, y_pred)
    return {
        "threshold": float(threshold),
        **metrics,
    }


def sweep_thresholds(
    y_true: Sequence[int],
    y_score: Sequence[float],
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float]]:
    if thresholds is None:
        try:
            import numpy as np

            raw = np.asarray(y_score, dtype=float)
            unique = np.unique(raw)
            if unique.size <= 4096:
                thresholds = sorted((float(value) for value in unique), reverse=True)
            else:
                thresholds = [float(value) for value in np.linspace(0.0, 1.0, 2001)]
        except Exception:  # pragma: no cover - numpy-free fallback
            thresholds = [index / 1000.0 for index in range(1000, -1, -1)]

    rows = [
        evaluate_threshold(y_true, y_score, float(threshold))
        for threshold in thresholds
    ]
    rows.sort(key=lambda row: float(row["threshold"]))
    return rows


def select_best_row_by_max_fpr(
    rows: Sequence[dict[str, float]], max_fpr: float
) -> dict[str, float]:
    candidates = [row for row in rows if float(row.get("fpr", 1.0)) <= max_fpr]
    if not candidates:
        candidates = list(rows)
        return min(
            candidates,
            key=lambda row: (
                float(row.get("fpr", 1.0)),
                -float(row.get("recall", 0.0)),
                -float(row.get("precision", 0.0)),
                -float(row.get("threshold", 0.0)),
            ),
        )

    return max(
        candidates,
        key=lambda row: (
            float(row.get("recall", 0.0)),
            float(row.get("precision", 0.0)),
            -float(row.get("fpr", 1.0)),
            float(row.get("threshold", 0.0)),
        ),
    )


def select_best_row_by_max_fpr_max_precision(
    rows: Sequence[dict[str, float]], max_fpr: float
) -> dict[str, float]:
    candidates = [row for row in rows if float(row.get("fpr", 1.0)) <= max_fpr]
    if not candidates:
        candidates = list(rows)
        return min(
            candidates,
            key=lambda row: (
                float(row.get("fpr", 1.0)),
                -float(row.get("precision", 0.0)),
                -float(row.get("recall", 0.0)),
                -float(row.get("threshold", 0.0)),
            ),
        )

    return max(
        candidates,
        key=lambda row: (
            float(row.get("precision", 0.0)),
            float(row.get("recall", 0.0)),
            -float(row.get("fpr", 1.0)),
            float(row.get("threshold", 0.0)),
        ),
    )


def select_best_row_by_min_recall(
    rows: Sequence[dict[str, float]], min_recall: float
) -> dict[str, float]:
    candidates = [row for row in rows if float(row.get("recall", 0.0)) >= min_recall]
    if not candidates:
        candidates = list(rows)
        return max(
            candidates,
            key=lambda row: (
                float(row.get("recall", 0.0)),
                -float(row.get("fpr", 1.0)),
                float(row.get("precision", 0.0)),
                float(row.get("threshold", 0.0)),
            ),
        )

    return min(
        candidates,
        key=lambda row: (
            float(row.get("fpr", 1.0)),
            -float(row.get("precision", 0.0)),
            -float(row.get("recall", 0.0)),
            -float(row.get("threshold", 0.0)),
        ),
    )


def select_best_row_by_min_recall_max_precision(
    rows: Sequence[dict[str, float]], min_recall: float
) -> dict[str, float]:
    candidates = [row for row in rows if float(row.get("recall", 0.0)) >= min_recall]
    if not candidates:
        candidates = list(rows)
        return max(
            candidates,
            key=lambda row: (
                float(row.get("recall", 0.0)),
                float(row.get("precision", 0.0)),
                -float(row.get("fpr", 1.0)),
                float(row.get("threshold", 0.0)),
            ),
        )

    return max(
        candidates,
        key=lambda row: (
            float(row.get("precision", 0.0)),
            -float(row.get("fpr", 1.0)),
            float(row.get("recall", 0.0)),
            float(row.get("threshold", 0.0)),
        ),
    )


def split_inner_train_calibration(
    indices: Sequence[int],
    labels: Sequence[int],
    calibration_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    grouped: dict[int, list[int]] = {}
    for idx in indices:
        grouped.setdefault(int(labels[idx]), []).append(int(idx))

    rng = random.Random(seed)
    train_idx: list[int] = []
    calib_idx: list[int] = []
    for label in sorted(grouped):
        bucket = grouped[label][:]
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            train_idx.extend(bucket)
            continue

        calibration_n = int(round(len(bucket) * calibration_fraction))
        calibration_n = max(1, calibration_n)
        calibration_n = min(len(bucket) - 1, calibration_n)

        calib_idx.extend(bucket[:calibration_n])
        train_idx.extend(bucket[calibration_n:])

    return sorted(train_idx), sorted(calib_idx)


def choose_stage_layers(
    rows: Sequence[dict[str, float]],
    early_fraction: float = 0.5,
    late_fraction: float = 0.5,
    early_metric: str = "recall_at_fpr_0_001",
    late_metric: str = "precision_at_fpr_0_0001",
) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty")

    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["layer"]))
    count = len(ordered)
    early_count = max(1, int(math.ceil(count * early_fraction)))
    late_count = max(1, int(math.ceil(count * late_fraction)))

    early_candidates = ordered[:early_count]
    late_candidates = ordered[-late_count:]

    early = max(
        early_candidates,
        key=lambda row: (
            float(row.get(early_metric, 0.0)),
            -int(row["layer"]),
        ),
    )

    late_pool = late_candidates[:]
    if len(late_pool) > 1:
        late_pool = [
            row for row in late_pool if int(row["layer"]) != int(early["layer"])
        ] or late_pool
    late = max(
        late_pool,
        key=lambda row: (
            float(row.get(late_metric, 0.0)),
            int(row["layer"]),
        ),
    )

    return {
        "early": dict(early),
        "late": dict(late),
        "early_candidates": [dict(row) for row in early_candidates],
        "late_candidates": [dict(row) for row in late_candidates],
        "early_metric": early_metric,
        "late_metric": late_metric,
    }


def apply_cascade(
    stage1_scores: Sequence[float],
    stage2_scores: Sequence[float],
    stage1_threshold: float,
    stage2_threshold: float,
) -> list[int]:
    if len(stage1_scores) != len(stage2_scores):
        raise ValueError("stage1_scores and stage2_scores must have the same length")

    preds = []
    for stage1_score, stage2_score in zip(stage1_scores, stage2_scores):
        passed_stage1 = float(stage1_score) >= stage1_threshold
        passed_stage2 = float(stage2_score) >= stage2_threshold
        preds.append(1 if (passed_stage1 and passed_stage2) else 0)
    return preds


def false_positives_per_million(fp: int | float, n_negative: int | float) -> float:
    if not n_negative:
        return 0.0
    return float(fp) * 1_000_000.0 / float(n_negative)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)
