from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-folds", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    summary_rows = []
    fold_rows = []
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_csv = model_dir / "summary.csv"
        fold_csv = model_dir / "fold_metrics.csv"
        if summary_csv.exists():
            summary_rows.extend(read_csv_rows(summary_csv))
        if fold_csv.exists():
            fold_rows.extend(read_csv_rows(fold_csv))

    if summary_rows:
        write_csv_rows(
            Path(args.output_summary), summary_rows, list(summary_rows[0].keys())
        )
    if fold_rows:
        write_csv_rows(Path(args.output_folds), fold_rows, list(fold_rows[0].keys()))

    print(f"summary_rows={len(summary_rows)}")
    print(f"fold_rows={len(fold_rows)}")


if __name__ == "__main__":
    main()
