#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "BTV-like": "#315b66",
    "HK97-like": "#8fa8a4",
    "NCLDV-like": "#d4a373",
    "picorna-like": "#8d99ae",
    "micro-like": "#b8c0a8",
    "ino-like": "#BE185D",
    "Circoviridae-like": "#c7b8a3",
    "Geminiviridae-like": "#b8a4b3",
    "levi-like": "#a7adb5",
    "Unknown": "#9ca3af",
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render one Figure S5 panel.")
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "plots")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    plots_dir = args.output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    if args.panel == "A":
        coords = pd.read_csv(data_dir / "panel_a_embedding_coordinates_input.csv")
        fig, ax = plt.subplots(figsize=(7.0, 6.8))
        for group, rows in coords.groupby("fold", sort=False):
            color = PALETTE.get(str(group), "#b0b0b0")
            is_btv = group == "BTV-like"
            is_unknown = group == "Unknown"
            ax.scatter(
                rows["x"], rows["y"],
                s=42 if is_btv else (30 if is_unknown else 22),
                c=color,
                edgecolors="#1f4149" if is_btv else ("#666666" if is_unknown else "white"),
                linewidths=0.8 if is_btv else 0.45,
                alpha=0.95,
                label=group,
            )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(color="#ececec", linewidth=0.7)
        legend_order = [
            "BTV-like", "HK97-like", "NCLDV-like", "picorna-like", "micro-like",
            "ino-like", "Geminiviridae-like", "Circoviridae-like", "levi-like",
        ]
        handles, labels = ax.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles, strict=True))
        ordered_labels = [label for label in legend_order if label in handle_by_label]
        ax.legend(
            [handle_by_label[label] for label in ordered_labels],
            ordered_labels,
            title="Capsid architecture",
            fontsize=7,
            title_fontsize=8,
            frameon=False,
            loc="best",
            ncol=2,
        )
        stem = "panel_a_embedding_reproduced"
    else:
        summary = pd.read_csv(data_dir / "panel_b_position_divergence.csv")
        plot_summary = summary.sort_values("mean_jensen_shannon_divergence")
        y = np.arange(len(plot_summary))
        colors = [PALETTE.get(str(x), "#315b66") for x in plot_summary["fold_group"]]
        fig, ax = plt.subplots(figsize=(7.0, 6.8))
        ax.barh(y, plot_summary["mean_jensen_shannon_divergence"], color=colors, alpha=0.88)
        ax.set_yticks(y, plot_summary["fold_group"])
        ax.set_xlabel("Mean pairwise Jensen–Shannon divergence")
        ax.grid(axis="x", color="#ececec", linewidth=0.7)
        ax.set_axisbelow(True)
        stem = "panel_b_divergence_reproduced"

    fig.tight_layout()
    fig.savefig(plots_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
