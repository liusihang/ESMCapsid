#!/usr/bin/env python3
"""Render editable vector outputs for ESMCapsid-C HDBSCAN scan metrics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "hdbscan_scan_metrics.csv"
TEXT = "#111827"
MUTED = "#6B7280"
GRID = "#D1D5DB"
BEST = "#B91C1C"

COMBO_COLORS = {
    (99, 50): "#1D4ED8",
    (149, 50): "#0F766E",
    (199, 50): "#7C3AED",
    (249, 50): "#B45309",
    (249, 200): "#DC2626",
}

METRIC_SPECS = [
    (
        "mean_persistence",
        "mean_persistence across all scanned parameter combinations",
    ),
    (
        "percent_noise",
        "percent_noise across all scanned parameter combinations",
    ),
]


def configure_vector_text_exports() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def fmt(value: float) -> str:
    return f"{value:.3f}"


def load_rows(path: Path) -> tuple[list[dict[str, object]], list[int], list[tuple[int, int]]]:
    rows = []
    layers = set()
    combos = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            combo = (int(row["min_cluster_size"]), int(row["min_samples"]))
            parsed = {
                "layer": row["layer"],
                "layer_num": int(row["layer_num"]),
                "min_cluster_size": combo[0],
                "min_samples": combo[1],
                "percent_noise": float(row["percent_noise"]),
                "mean_persistence": float(row["mean_persistence"]),
                "is_best_for_layer": row["is_best_for_layer"] == "True",
            }
            rows.append(parsed)
            layers.add(parsed["layer_num"])
            combos.add(combo)
    return rows, sorted(layers), sorted(combos)


def build_legend_handles(combos: list[tuple[int, int]]) -> list[Line2D]:
    handles = []
    for combo in combos:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COMBO_COLORS[combo],
                linewidth=2.4,
                marker="o",
                markersize=6,
                markerfacecolor=COMBO_COLORS[combo],
                markeredgecolor="white",
                label=f"mcs={combo[0]}, min_samples={combo[1]}",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgewidth=2,
            markeredgecolor=BEST,
            label="best row for layer",
        )
    )
    return handles


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", linewidth=0.7, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)
    ax.tick_params(colors=TEXT)


def plot_metric_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    layers: list[int],
    combos: list[tuple[int, int]],
    metric_key: str,
    title: str,
) -> None:
    style_axis(ax)
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["min_cluster_size"]), int(row["min_samples"]))].append(row)

    values = [float(row[metric_key]) for row in rows]
    vmin = min(values)
    vmax = max(values)
    pad = (vmax - vmin) * 0.12 if vmax > vmin else 0.01
    ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    for combo in combos:
        combo_rows = sorted(grouped[combo], key=lambda row: int(row["layer_num"]))
        xs = [int(row["layer_num"]) for row in combo_rows]
        ys = [float(row[metric_key]) for row in combo_rows]
        ax.plot(
            xs,
            ys,
            color=COMBO_COLORS[combo],
            linewidth=2.4,
            marker="o",
            markersize=6,
            markerfacecolor=COMBO_COLORS[combo],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        best_rows = [row for row in combo_rows if bool(row["is_best_for_layer"])]
        if best_rows:
            ax.scatter(
                [int(row["layer_num"]) for row in best_rows],
                [float(row[metric_key]) for row in best_rows],
                s=110,
                facecolors="none",
                edgecolors=BEST,
                linewidths=2,
                zorder=4,
            )

    ax.set_title(title, fontsize=12, color=TEXT, pad=8)
    ax.set_xlabel("Layer", color=TEXT)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{layer}" for layer in layers], rotation=0)


def build_panel(metric_key: str) -> plt.Figure:
    configure_vector_text_exports()
    rows, layers, combos = load_rows(INPUT_CSV)

    metric_title = dict(METRIC_SPECS)[metric_key]
    fig, ax = plt.subplots(figsize=(12.6, 4.8), dpi=300)
    fig.subplots_adjust(top=0.80, left=0.09, right=0.98, bottom=0.16)
    handles = build_legend_handles(combos)
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.09, 0.97),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=2.4,
        columnspacing=1.4,
    )
    plot_metric_panel(ax, rows, layers, combos, metric_key, metric_title)
    ax.set_ylabel(metric_key, color=TEXT)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one Figure S3 metric panel.")
    parser.add_argument("--metric", choices=tuple(key for key, _ in METRIC_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    args = parser.parse_args()
    fig = build_panel(args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"panel_{args.metric}_reproduced"
    try:
        fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    finally:
        plt.close(fig)
    print(prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
