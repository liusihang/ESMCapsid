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
    parser.add_argument("--output-level-summary", required=True)
    parser.add_argument("--output-per-taxon", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    level_rows = []
    per_taxon_rows = []

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("_"):
            continue
        level_summary = model_dir / "level_summary.csv"
        per_taxon = model_dir / "per_taxon_metrics.csv"
        if level_summary.exists():
            level_rows.extend(read_csv_rows(level_summary))
        if per_taxon.exists():
            per_taxon_rows.extend(read_csv_rows(per_taxon))

    if level_rows:
        write_csv_rows(
            Path(args.output_level_summary), level_rows, list(level_rows[0].keys())
        )
    if per_taxon_rows:
        write_csv_rows(
            Path(args.output_per_taxon), per_taxon_rows, list(per_taxon_rows[0].keys())
        )

    print(f"level_rows={len(level_rows)}")
    print(f"per_taxon_rows={len(per_taxon_rows)}")


if __name__ == "__main__":
    main()
