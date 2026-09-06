from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues


ROOT = Path("/path/to/science_workspace")
INPUT_DIR = ROOT / "pdb_capsid_assemblies" / "group_core_motif_mapping_20260307"

ARCH_FILE = INPUT_DIR / "updated_core_architecture_metrics_per_structure.csv"
INTERFACE_FILE = INPUT_DIR / "updated_core_interface_metrics_per_structure.csv"

SUMMARY_OUT = INPUT_DIR / "updated_cluster_core_quantitative_evidence_summary.csv"
DETAIL_OUT = INPUT_DIR / "updated_cluster_core_quantitative_evidence_per_structure.csv"
FIG_OUT = INPUT_DIR / "pngs" / "updated_cluster_core_quantitative_evidence.png"

GROUP_ORDER = ["HK97_like", "NCLDV_like", "Pico_like"]
GROUP_COLORS = {
    "HK97_like": "#1f4e79",
    "NCLDV_like": "#8c2d04",
    "Pico_like": "#2b6c4a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-structure cluster-core quantitative evidence."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    return parser.parse_args()


def fisher_pvalue(values: pd.Series) -> float:
    clean = values.dropna().clip(lower=1e-300)
    if clean.empty:
        return float("nan")
    return float(combine_pvalues(clean, method="fisher").pvalue)


def prepare_detail_table(
    arch_df: pd.DataFrame, interface_df: pd.DataFrame
) -> pd.DataFrame:
    merged = arch_df.merge(
        interface_df,
        on=["group", "seq_id", "n_core_res", "chain_len"],
        how="left",
        suffixes=("", "_interface"),
    ).copy()

    merged["radial_std_ratio"] = (
        merged["core_radial_std"] / merged["rand_radial_std_mean"]
    )
    merged["pairwise_ratio"] = (
        merged["core_pairwise_median"] / merged["rand_pairwise_median_mean"]
    )
    merged["neighbor10_ratio"] = (
        merged["core_neighbor10_mean"] / merged["rand_neighbor10_mean"]
    )
    merged["otherchain_minCA_delta"] = (
        merged["core_mean_otherchain_minCA"] - merged["bg_mean_otherchain_minCA"]
    )
    merged["interface_frac_delta"] = (
        merged["core_frac_interface_lt8A"] - merged["bg_frac_interface_lt8A"]
    )
    merged["neighbors12_ratio"] = (
        merged["core_mean_neighbors12A"] / merged["bg_mean_neighbors12A"]
    )
    merged["radial_scaled_delta"] = (
        merged["core_mean_radial_scaled"] - merged["bg_mean_radial_scaled"]
    )
    return merged


def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in GROUP_ORDER:
        group_df = detail_df[detail_df["group"] == group].copy()
        if group_df.empty:
            continue

        rows.append(
            {
                "group": group,
                "n_structures": int(len(group_df)),
                "mean_n_core_res": float(group_df["n_core_res"].mean()),
                "mean_core_frac": float(group_df["core_frac"].mean()),
                "mean_radial_std_ratio": float(group_df["radial_std_ratio"].mean()),
                "mean_radial_std_reduction_pct": float(
                    (1.0 - group_df["radial_std_ratio"]).mean() * 100.0
                ),
                "mean_radial_std_z": float(group_df["radial_std_z"].mean()),
                "fisher_radial_p": fisher_pvalue(
                    group_df["radial_std_empirical_p_lower"]
                ),
                "n_radial_p_lt_0_05": int(
                    (group_df["radial_std_empirical_p_lower"] < 0.05).sum()
                ),
                "mean_pairwise_ratio": float(group_df["pairwise_ratio"].mean()),
                "mean_pairwise_reduction_pct": float(
                    (1.0 - group_df["pairwise_ratio"]).mean() * 100.0
                ),
                "mean_pairwise_z": float(group_df["pairwise_z"].mean()),
                "fisher_pairwise_p": fisher_pvalue(
                    group_df["pairwise_empirical_p_lower"]
                ),
                "n_pairwise_p_lt_0_05": int(
                    (group_df["pairwise_empirical_p_lower"] < 0.05).sum()
                ),
                "mean_neighbor10_ratio": float(group_df["neighbor10_ratio"].mean()),
                "mean_neighbor10_increase_pct": float(
                    (group_df["neighbor10_ratio"] - 1.0).mean() * 100.0
                ),
                "mean_neighbor10_z": float(group_df["neighbor10_z"].mean()),
                "fisher_neighbor10_p": fisher_pvalue(
                    group_df["neighbor10_empirical_p_higher"]
                ),
                "n_neighbor10_p_lt_0_05": int(
                    (group_df["neighbor10_empirical_p_higher"] < 0.05).sum()
                ),
                "mean_otherchain_minCA_delta": float(
                    group_df["otherchain_minCA_delta"].mean()
                ),
                "mean_interface_frac_delta": float(
                    group_df["interface_frac_delta"].mean()
                ),
                "n_structures_core_interface_gt_bg": int(
                    (group_df["interface_frac_delta"] > 0).sum()
                ),
                "mean_neighbors12_ratio": float(group_df["neighbors12_ratio"].mean()),
                "n_structures_neighbors12_gt_bg": int(
                    (group_df["neighbors12_ratio"] > 1).sum()
                ),
                "mean_radial_scaled_delta": float(
                    group_df["radial_scaled_delta"].mean()
                ),
                "n_structures_radial_more_inner": int(
                    (group_df["radial_scaled_delta"] < 0).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_metric_panel(
    ax: plt.Axes,
    detail_df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    positive_is_enriched: bool,
) -> None:
    rng = np.random.default_rng(7)
    for idx, group in enumerate(GROUP_ORDER):
        group_df = detail_df[detail_df["group"] == group]
        if group_df.empty:
            continue

        y = np.full(len(group_df), idx, dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(group_df))
        color = GROUP_COLORS[group]
        ax.scatter(
            group_df[column],
            y + jitter,
            s=48,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter(
            [group_df[column].mean()],
            [idx],
            marker="D",
            s=92,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels(GROUP_ORDER)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)

    if positive_is_enriched:
        ax.annotate(
            "more than random",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#444444",
        )
    else:
        ax.annotate(
            "more constrained than random",
            xy=(0.01, 0.02),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#444444",
        )


def make_figure(detail_df: pd.DataFrame) -> None:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    plot_metric_panel(
        axes[0, 0],
        detail_df,
        column="radial_std_z",
        title="Radial Constraint",
        xlabel="z-score of radial spread",
        positive_is_enriched=False,
    )
    plot_metric_panel(
        axes[0, 1],
        detail_df,
        column="pairwise_z",
        title="3D Spatial Clustering",
        xlabel="z-score of pairwise distance",
        positive_is_enriched=False,
    )
    plot_metric_panel(
        axes[1, 0],
        detail_df,
        column="neighbor10_z",
        title="Local Packing Density",
        xlabel="z-score of 10 A neighbor count",
        positive_is_enriched=True,
    )
    plot_metric_panel(
        axes[1, 1],
        detail_df,
        column="interface_frac_delta",
        title="Direct Interface Enrichment",
        xlabel="core minus background fraction (<8 A to other chain)",
        positive_is_enriched=True,
    )

    fig.suptitle(
        "Cluster-level lineage core motifs: quantitative structural evidence",
        fontsize=15,
        y=1.02,
    )
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global INPUT_DIR, ARCH_FILE, INTERFACE_FILE
    global SUMMARY_OUT, DETAIL_OUT, FIG_OUT

    args = parse_args()
    INPUT_DIR = args.input_dir.expanduser().resolve()
    ARCH_FILE = INPUT_DIR / "updated_core_architecture_metrics_per_structure.csv"
    INTERFACE_FILE = INPUT_DIR / "updated_core_interface_metrics_per_structure.csv"
    SUMMARY_OUT = INPUT_DIR / "updated_cluster_core_quantitative_evidence_summary.csv"
    DETAIL_OUT = (
        INPUT_DIR / "updated_cluster_core_quantitative_evidence_per_structure.csv"
    )
    FIG_OUT = INPUT_DIR / "pngs" / "updated_cluster_core_quantitative_evidence.png"

    arch_df = pd.read_csv(ARCH_FILE)
    interface_df = pd.read_csv(INTERFACE_FILE)

    detail_df = prepare_detail_table(arch_df, interface_df)
    summary_df = build_summary(detail_df)

    detail_df.to_csv(DETAIL_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)
    make_figure(detail_df)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False))
    print(f"\nWrote {DETAIL_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
