#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG4_ROOT = Path(__file__).resolve().parents[1]
TOKEN_STATS = FIG4_ROOT / "data" / "panel_a_token_distribution.csv"
HK97_POSITION_BINS = (
    FIG4_ROOT / "data" / "panel_c_hk97_position_profile.csv"
)
PICO_POSITION_BINS = (
    FIG4_ROOT / "data" / "panel_c_picorna_position_profile.csv"
)
SCAFFOLD_METRICS = (
    FIG4_ROOT / "data" / "panel_e_scaffold_metrics.csv"
)
COLORS = {
    "core": "#1F4E79",
    "driver": "#B55A42",
    "background": "#A7B3C7",
    "hk97": "#1F4E79",
    "pico": "#2B6C4A",
    "secondary": "#A66A4C",
    "grid": "#D8DDE6",
    "text": "#222222",
}

TOKEN_CATEGORY_COLORS = {
    "Dominating (Low Complexity)": "#D95F5F",
    "Random / Scattered": "#6EA6D7",
    "N-Terminal": "#79C47E",
    "C-Terminal": "#B176B8",
    "Middle (Conserved)": "#F4A340",
}
TOKEN_CATEGORY_LABELS = {
    "Dominating (Low Complexity)": "Dominating in specific proteins",
    "Middle (Conserved)": "Middle",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["text"],
            "axes.linewidth": 1.5,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str | None = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.2, length=4, color=COLORS["text"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.65)
    else:
        ax.grid(False)


def plot_token_distribution(ax: plt.Axes, stats_path: Path = TOKEN_STATS) -> None:
    df = pd.read_csv(stats_path)
    size_min, size_max = 28, 230
    mean_fraction = df["mean_fraction"].fillna(0).clip(lower=0)
    if mean_fraction.max() > mean_fraction.min():
        sizes = size_min + (mean_fraction - mean_fraction.min()) / (
            mean_fraction.max() - mean_fraction.min()
        ) * (size_max - size_min)
    else:
        sizes = np.full(len(df), 80.0)

    for category, sub in df.groupby("position_category", sort=False):
        idx = sub.index
        ax.scatter(
            sub["mean_relative_position"],
            sub["position_standard_deviation"],
            s=sizes.loc[idx],
            color=TOKEN_CATEGORY_COLORS.get(category, "#777777"),
            alpha=0.72,
            edgecolors="white",
            linewidth=0.5,
            label=TOKEN_CATEGORY_LABELS.get(category, category),
            zorder=3,
        )

    ax.axhline(
        0.288,
        color="#777777",
        linestyle="--",
        linewidth=1.5,
        label="Uniform dist. std",
        zorder=2,
    )
    ax.set_xlabel("Average relative position (0=N, 1=C)")
    ax.set_ylabel("Standard deviation of position")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.015, max(0.36, float(df["position_standard_deviation"].max()) * 1.08))
    style_axis(ax)
    legend = ax.legend(
        title="Distribution",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        borderpad=0.8,
        labelspacing=0.55,
        handletextpad=0.5,
    )
    legend._legend_box.align = "left"


def plot_position_profile(ax: plt.Axes, csv_path: Path, title: str, show_legend: bool = False) -> None:
    df = pd.read_csv(csv_path)
    x = (df["bin_start"] + df["bin_end"]) / 2
    ax.plot(
        x,
        df["frac_background_all_tokens"],
        color=COLORS["background"],
        linewidth=2.0,
        label="Background",
        zorder=2,
    )
    ax.plot(
        x,
        df["frac_core_motifs"],
        color=COLORS["core"],
        linewidth=2.0,
        label="Core motifs",
        zorder=3,
    )
    ax.plot(
        x,
        df["frac_driver_motifs"],
        color=COLORS["driver"],
        linewidth=2.0,
        label="Driver motifs",
        zorder=3,
    )
    ax.set_title(title, pad=4, fontsize=11)
    ax.set_xlabel("Relative sequence position (0 -> 1)", fontsize=10)
    ax.set_ylabel("Fraction of occurrences", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    style_axis(ax)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=8)


def load_scaffold_metrics(path: Path = SCAFFOLD_METRICS) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "radial_std_z",
        "pairwise_z",
        "neighbor10_z",
        "interface_frac_delta",
        "band_cross_z",
        "anisotropy_z",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_scaffold_metric_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    *,
    secondary_col: str | None = None,
    secondary_label: str | None = None,
    show_legend: bool = False,
) -> None:
    labels = df["display_label"].astype(str).tolist()
    x_all = np.arange(len(df))
    for group, sub in df.groupby("group", sort=False):
        positions = np.array([df.index.get_loc(idx) for idx in sub.index], dtype=float)
        ax.scatter(
            positions,
            sub[y_col],
            s=44,
            color=COLORS["hk97"] if group == "HK97_like" else COLORS["pico"],
            edgecolors="white",
            linewidth=0.5,
            alpha=0.78,
            label="HK97-like" if group == "HK97_like" else "Pico-like",
            zorder=3,
        )

    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_title(title, pad=5, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    style_axis(ax, grid_axis="y")
    if show_legend:
        ax.legend(frameon=False, loc="lower left", fontsize=8)

    if secondary_col:
        ax2 = ax.twinx()
        ax2.plot(
            x_all,
            df[secondary_col],
            color=COLORS["secondary"],
            marker="D",
            markersize=3.4,
            linewidth=1.4,
            alpha=0.9,
            zorder=2,
        )
        ax2.set_ylabel(secondary_label, color=COLORS["secondary"], fontsize=8)
        ax2.tick_params(axis="y", colors=COLORS["secondary"], labelsize=8, width=1.0)
        ax2.spines["right"].set_color(COLORS["secondary"])
        ax2.spines["right"].set_linewidth(1.2)
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)


def plot_scaffold_evidence(axes: Iterable[plt.Axes], metrics_path: Path = SCAFFOLD_METRICS) -> None:
    df = load_scaffold_metrics(metrics_path)
    axes = list(axes)
    plot_scaffold_metric_panel(
        axes[0],
        df,
        "radial_std_z",
        "Shell restriction",
        "Radial z",
        show_legend=True,
    )
    plot_scaffold_metric_panel(axes[1], df, "pairwise_z", "Compactness", "Pairwise z")
    plot_scaffold_metric_panel(
        axes[2],
        df,
        "neighbor10_z",
        "Contact enrichment",
        "Neighbor z",
        secondary_col="interface_frac_delta",
        secondary_label="Interface delta",
    )
    plot_scaffold_metric_panel(
        axes[3],
        df,
        "band_cross_z",
        "Recurrent geometry",
        "Bandness z",
        secondary_col="anisotropy_z",
        secondary_label="Anisotropy z",
    )


def save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one Figure 4 panel at a time.")
    parser.add_argument("--panel", choices=("A", "C", "E"), required=True)
    parser.add_argument("--output-dir", type=Path, default=FIG4_ROOT / "plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    if args.panel == "A":
        fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=300)
        plot_token_distribution(ax)
        stem = "panel_a_token_distribution_reproduced"
    elif args.panel == "C":
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), dpi=300, sharex=True)
        plot_position_profile(axes[0], HK97_POSITION_BINS, "HK97-like", show_legend=True)
        plot_position_profile(axes[1], PICO_POSITION_BINS, "Pico-like")
        stem = "panel_c_position_profiles_reproduced"
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=300)
        plot_scaffold_evidence(axes.flat, metrics_path=SCAFFOLD_METRICS)
        stem = "panel_e_scaffold_evidence_reproduced"
    save_panel(fig, args.output_dir, stem)
    print(args.output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
