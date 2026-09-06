from __future__ import annotations

import csv
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SPLIT_CSV = ROOT / "data/panel_a_random_split.csv"
LOW_IDENTITY_SUMMARY_CSV = ROOT / "data/panel_a_low_identity.csv"
TAXONOMY_LEVEL_SUMMARY_CSV = ROOT / "data/panel_a_taxonomy_holdout.csv"
C_CSV = ROOT / "data/panel_c_mlm_metrics.csv"
D_CSV = ROOT / "data/panel_d_downstream_metrics.csv"

MODEL_ORDER_A = [
    "ESMplusplus_large",
    "esm3o",
    "Profluent-E1-600M",
    "FastESM2_650M",
    "ESMCapsid-S",
]
MODEL_LABELS_A = {
    "ESMplusplus_large": "ESM++",
    "esm3o": "ESM3",
    "Profluent-E1-600M": "Profluent",
    "FastESM2_650M": "FastESM2",
    "ESMCapsid-S": "ESMCapsid-S",
}
MODEL_ORDER_C = [
    "ESMplusplus_large",
    "esm3o",
    "Profluent-E1-600M",
    "FastESM2_650M",
    "ESMCapsid-C",
]
MODEL_LABELS_C = {
    "ESMplusplus_large": "ESM++",
    "esm3o": "ESM3",
    "Profluent-E1-600M": "Profluent",
    "FastESM2_650M": "FastESM2",
    "ESMCapsid-C": "ESMCapsid-C",
}

COLORS_A = {
    "ESMplusplus_large": "#8D99AE",
    "esm3o": "#6D7A8C",
    "Profluent-E1-600M": "#B98B73",
    "FastESM2_650M": "#7AA6A1",
    "ESMCapsid-S": "#C84C31",
}
COLORS_C = {
    "Capsid": "#C84C31",
    "Non-capsid virus": "#327A88",
    "Others / Cellular proteins": "#C59A3D",
}
COLOR_D_PRE = "#B8C2C9"
COLOR_D_POST = "#C84C31"
FIG_BG = "#FFFFFF"
AX_BG = "#FFFFFF"
GRID = "#E6E0D8"

A_METRIC_ORDER = ["Capsid F1", "Balanced Acc.", "Capsid Recall"]
A_METRIC_COLORS = {
    "Capsid F1": "#7A4E6A",
    "Balanced Acc.": "#5E6C7A",
    "Capsid Recall": "#95864B",
}
A_METRIC_LABELS = {
    "Capsid F1": "F1",
    "Balanced Acc.": "Balanced accuracy",
    "Capsid Recall": "Recall",
}
def configure_vector_text_exports() -> None:
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_a_metric_triples():
    data = defaultdict(lambda: defaultdict(dict))

    with RANDOM_SPLIT_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            model = row["Model"]
            if model not in MODEL_ORDER_A:
                continue
            data["Random split"][model]["Capsid F1"] = float(row["Capsid F1"])
            data["Random split"][model]["Balanced Acc."] = float(row["Balanced Acc."])
            data["Random split"][model]["Capsid Recall"] = float(row["Capsid Recall"])

    with LOW_IDENTITY_SUMMARY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if model not in MODEL_ORDER_A:
                continue
            data["30% low-identity"][model]["Capsid F1"] = float(row["capsid_f1_mean"])
            data["30% low-identity"][model]["Balanced Acc."] = float(row["balanced_accuracy_mean"])
            data["30% low-identity"][model]["Capsid Recall"] = float(row["capsid_recall_mean"])

    with TAXONOMY_LEVEL_SUMMARY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["level"] != "Family":
                continue
            model = row["model"]
            if model not in MODEL_ORDER_A:
                continue
            data["Family-held-out"][model]["Capsid F1"] = float(row["capsid_f1_mean"])
            data["Family-held-out"][model]["Balanced Acc."] = float(row["balanced_accuracy_mean"])
            data["Family-held-out"][model]["Capsid Recall"] = float(row["capsid_recall_mean"])

    return data


def load_c():
    data = defaultdict(dict)
    with C_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] not in {"Accuracy", "Perplexity"}:
                continue
            key = (row["metric"], row["sequence_group"])
            data[key][row["model_display"]] = float(row["mean"])
    return data


def load_d():
    rows = []
    with D_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] != "Macro F1":
                continue
            rows.append(
                (
                    row["level"],
                    float(row["pre_finetuning"]),
                    float(row["post_finetuning"]),
                    float(row["diff"]),
                )
            )
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A9A39B")
    ax.spines["bottom"].set_color("#A9A39B")
    ax.set_facecolor(AX_BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors="#4B4742")


def draw_a(ax, title, values, *, show_legend=False):
    offsets = [-0.26, 0.0, 0.26]
    xs = list(range(len(MODEL_ORDER_A)))
    for offset, metric in zip(offsets, A_METRIC_ORDER):
        ys = [values[m][metric] for m in MODEL_ORDER_A]
        ax.bar(
            [x + offset for x in xs],
            ys,
            color=A_METRIC_COLORS[metric],
            width=0.24,
            label=A_METRIC_LABELS[metric],
        )
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS_A[m] for m in MODEL_ORDER_A], rotation=22, ha="right")
    ax.set_ylim(0.35 if "Family" in title else 0.74, 1.02)
    style_axis(ax)
    ax.set_ylabel("Capsid score", fontsize=8.5)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.2, loc="upper left", handlelength=1.1, labelspacing=0.4)


def draw_c_grouped(ax, metric, cdata):
    seq_groups = ["Capsid", "Non-capsid virus", "Others / Cellular proteins"]
    group_offsets = [-0.24, 0.0, 0.24]
    xs = list(range(len(MODEL_ORDER_C)))

    for offset, seq_group in zip(group_offsets, seq_groups):
        ys = [cdata[(metric, seq_group)][m] for m in MODEL_ORDER_C]
        yerr = [cdata[(metric, seq_group, "std")][m] for m in MODEL_ORDER_C]
        ax.bar(
            [x + offset for x in xs],
            ys,
            width=0.22,
            color=COLORS_C[seq_group],
            edgecolor="none",
            yerr=yerr,
            ecolor="#3E3A36",
            capsize=1.8,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
            label=seq_group,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS_C[m] for m in MODEL_ORDER_C], rotation=18, ha="right")
    style_axis(ax)
    if metric == "Accuracy":
        ax.set_ylim(0, 0.72)
        ax.set_ylabel("MLM accuracy", fontsize=8.5, color="#4B4742")
        ax.set_title("Accuracy", fontsize=10, pad=6, color="#2F2A25")
    else:
        ax.set_ylim(0, 24)
        ax.set_ylabel("Perplexity", fontsize=8.5, color="#4B4742")
        ax.set_title("Perplexity", fontsize=10, pad=6, color="#2F2A25")

    handles = [
        Patch(facecolor=COLORS_C["Capsid"], label="Capsid"),
        Patch(facecolor=COLORS_C["Non-capsid virus"], label="Non-capsid virus protein"),
        Patch(facecolor=COLORS_C["Others / Cellular proteins"], label="Others (cellular protein)"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.1,
        loc="upper left",
        handlelength=1.1,
        labelspacing=0.45,
    )


def draw_d(ax, rows):
    levels = [r[0] for r in rows]
    pre = [r[1] for r in rows]
    post = [r[2] for r in rows]
    y = list(range(len(levels)))
    for yi, p0, p1 in zip(y, pre, post):
        ax.plot([p0, p1], [yi, yi], color="#CBC4BB", linewidth=2.2, zorder=1)
    ax.scatter(pre, y, color=COLOR_D_PRE, s=28, zorder=3, label="Pre")
    ax.scatter(post, y, color=COLOR_D_POST, s=32, zorder=4, label="Post")
    ax.set_yticks(y)
    ax.set_yticklabels(levels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.78, 1.0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#A9A39B")
    ax.set_facecolor(AX_BG)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8, colors="#4B4742")
    ax.set_xlabel("Macro F1", fontsize=8.5, color="#4B4742")
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def load_c_with_std():
    c = load_c()
    c_with_std = {}
    with C_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] not in {"Accuracy", "Perplexity"}:
                continue
            c_with_std[(row["metric"], row["sequence_group"], "std")] = c_with_std.get(
                (row["metric"], row["sequence_group"], "std"), {}
            )
            c_with_std[(row["metric"], row["sequence_group"], "std")][row["model_display"]] = float(row["std"])
    c.update(c_with_std)
    return c


def save_panel(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        ("png", {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}),
        ("svg", {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}),
        ("pdf", {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}),
    ):
        fig.savefig(output_dir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one Figure 2 panel at a time.")
    parser.add_argument("--panel", choices=("A", "C", "D"), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    args = parser.parse_args()
    configure_vector_text_exports()

    if args.panel == "A":
        data = load_a_metric_triples()
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=320, facecolor=FIG_BG)
        for ax, key in zip(axes, ("Random split", "30% low-identity", "Family-held-out"), strict=True):
            draw_a(ax, key, data[key], show_legend=(key == "Random split"))
        stem = "panel_a_classification_reproduced"
    elif args.panel == "C":
        cdata = load_c_with_std()
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), dpi=320, facecolor=FIG_BG)
        draw_c_grouped(axes[0], "Accuracy", cdata)
        draw_c_grouped(axes[1], "Perplexity", cdata)
        stem = "panel_c_mlm_reproduced"
    else:
        fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=320, facecolor=FIG_BG)
        draw_d(ax, load_d())
        stem = "panel_d_downstream_reproduced"

    save_panel(fig, args.output_dir, stem)
    print(args.output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
