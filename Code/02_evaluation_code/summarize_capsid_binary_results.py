from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


POS_LABEL = "Capsid"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_prediction_file(predictions_path: Path):
    df = pd.read_csv(predictions_path)
    true_positive = (
        (df["true_label"] == POS_LABEL) & (df["pred_label"] == POS_LABEL)
    ).sum()
    false_positive = (
        (df["true_label"] != POS_LABEL) & (df["pred_label"] == POS_LABEL)
    ).sum()
    true_negative = (
        (df["true_label"] != POS_LABEL) & (df["pred_label"] != POS_LABEL)
    ).sum()
    false_negative = (
        (df["true_label"] == POS_LABEL) & (df["pred_label"] != POS_LABEL)
    ).sum()

    total = len(df)
    accuracy = safe_div(true_positive + true_negative, total)
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    specificity = safe_div(true_negative, true_negative + false_positive)
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "n_samples": int(total),
        "n_capsid_true": int((df["true_label"] == POS_LABEL).sum()),
        "n_capsid_pred": int((df["pred_label"] == POS_LABEL).sum()),
        "tp": int(true_positive),
        "fp": int(false_positive),
        "tn": int(true_negative),
        "fn": int(false_negative),
        "accuracy": accuracy,
        "precision_capsid": precision,
        "recall_capsid": recall,
        "specificity_non_capsid": specificity,
        "f1_capsid": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def load_multiclass_summary(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "multiclass_accuracy_mean": summary.get("accuracy_mean"),
        "multiclass_macro_f1_mean": summary.get("macro_f1_mean"),
        "multiclass_weighted_f1_mean": summary.get("weighted_f1_mean"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = []
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        predictions_path = model_dir / "predictions.csv"
        summary_path = model_dir / "summary.json"
        if not predictions_path.exists() or not summary_path.exists():
            continue

        row = {
            "model": model_dir.name,
            **load_multiclass_summary(summary_path),
            **summarize_prediction_file(predictions_path),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("f1_capsid", ascending=False)
    out_df.to_csv(args.output_csv, index=False)
    Path(args.output_json).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(out_df.to_csv(index=False))


if __name__ == "__main__":
    main()
