#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to render Figure S6.") from exc

    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render one Figure S6 ecosystem panel.")
    parser.add_argument("--ecosystem", choices=(
        "General Environmental",
        "Host Associated",
        "Extreme Environments",
        "Engineered Systems",
        "Contaminated and Industrial Environments",
        "Mixed And Lab",
    ), required=True)
    parser.add_argument("--output-dir", type=Path, default=base / "plots")
    args = parser.parse_args()

    source_csv = base / "data" / "cluster_accumulation.csv"
    plots_dir = args.output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not source_csv.is_file():
        raise SystemExit(f"Missing summarized plotting input: {source_csv}")

    with source_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "sample_size",
        "mean_cluster_recovery_fraction",
        "fraction_ci_low",
        "fraction_ci_high",
        "main_ecosystem",
        "n_sequences_total",
        "n_clusters_total",
    }
    missing_columns = required_columns.difference(rows[0] if rows else ())
    if missing_columns:
        raise SystemExit(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    colors = {
        "General Environmental": "#1f77b4",
        "Host Associated": "#d62728",
        "Extreme Environments": "#ff7f0e",
        "Engineered Systems": "#2ca02c",
        "Contaminated and Industrial Environments": "#9467bd",
        "Mixed And Lab": "#8c564b",
    }
    by_ecosystem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ecosystem[row["main_ecosystem"]].append(row)

    ecosystem = args.ecosystem
    ecosystem_rows = by_ecosystem.get(ecosystem, [])
    if not ecosystem_rows:
        raise SystemExit(f"No rows found for ecosystem: {ecosystem}")
    ecosystem_rows.sort(key=lambda row: int(row["sample_size"]))
    x_values = [int(row["sample_size"]) for row in ecosystem_rows]
    y_values = [float(row["mean_cluster_recovery_fraction"]) for row in ecosystem_rows]
    lower = [float(row["fraction_ci_low"]) for row in ecosystem_rows]
    upper = [float(row["fraction_ci_high"]) for row in ecosystem_rows]
    color = colors[ecosystem]

    figure, axis = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    axis.plot(x_values, y_values, color=color, linewidth=2)
    axis.fill_between(x_values, lower, upper, color=color, alpha=0.18)
    axis.set_xscale("log")
    axis.set_ylim(0, 1.02)
    axis.grid(True, which="both", alpha=0.25, linestyle="--", linewidth=0.6)
    axis.set_title(ecosystem, color=color, fontsize=11, pad=8)
    axis.set_xlabel("Sample size (sequences)")
    axis.set_ylabel("Observed cluster fraction (k / total clusters)")
    figure.tight_layout()
    slug = ecosystem.lower().replace(" ", "_")
    figure.savefig(plots_dir / f"panel_{slug}_reproduced.jpg", dpi=300, bbox_inches="tight")
    figure.savefig(plots_dir / f"panel_{slug}_reproduced.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
