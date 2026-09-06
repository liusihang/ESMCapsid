from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def configure_vector_text_exports() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


configure_vector_text_exports()


ROOT = Path(__file__).resolve().parents[1]
LAYERWISE_CSV = ROOT / "data/layer_probe_metrics.csv"
FINAL_CSV = ROOT / "data/two_head_metrics.csv"
OUT_DIR = ROOT / "plots"
OUT_PNG = OUT_DIR / "figure_s2_reproduced.png"
OUT_SVG = OUT_DIR / "figure_s2_reproduced.svg"
OUT_PDF = OUT_DIR / "figure_s2_reproduced.pdf"

FIG_BG = "#FFFFFF"
AX_BG = "#FFFFFF"
GRID = "#E6E0D8"
TEXT = "#1F1B18"
MUTED = "#4B4742"
SPINE = "#A9A39B"

COLOR_F1 = "#C84C31"
COLOR_ACC = "#327A88"
COLOR_REC = "#C59A3D"
COLOR_TIME = "#8D99AE"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.set_facecolor(AX_BG)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED)


def add_title(ax, label, title, subtitle):
    ax.text(-0.015, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", color=TEXT)
    ax.text(0.03, 1.08, title, transform=ax.transAxes, fontsize=11, va="top", color=TEXT)
    ax.text(0.03, 1.035, subtitle, transform=ax.transAxes, fontsize=8.2, va="top", color="#666666")


def load_layerwise():
    rows = []
    with LAYERWISE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "runtime": float(row["runtime_plot_minutes"]),
                }
            )
    rows.sort(key=lambda x: x["layer"])
    return rows


def load_final_two_stage():
    with FINAL_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] == "final_operating" and row["system"] == "two_stage_at_t1_t2":
                return {
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "fpr": float(row["fpr"]),
                    "note": row["threshold_or_rule"],
                }
    raise ValueError("Did not find final two-stage operating row in final CSV")


def main():
    layerwise = load_layerwise()
    final_row = load_final_two_stage()

    layers = [row["layer"] for row in layerwise]
    accuracies = [row["accuracy"] for row in layerwise]
    recalls = [row["recall"] for row in layerwise]
    f1s = [row["f1"] for row in layerwise]
    runtimes = [row["runtime"] for row in layerwise]
    layer16_runtime = next(row["runtime"] for row in layerwise if row["layer"] == 16)

    x_labels = [str(layer) for layer in layers] + ["L16+2H"]
    xs = list(range(len(x_labels)))
    metric_width = 0.22

    acc_all = accuracies + [final_row["accuracy"]]
    rec_all = recalls + [final_row["recall"]]
    f1_all = f1s + [final_row["f1"]]
    runtime_all = runtimes + [layer16_runtime]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(17.4, 7.8), dpi=320, facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)

    # Highlight the final pseudo-layer column.
    ax.axvspan(len(xs) - 1 - 0.5, len(xs) - 1 + 0.5, color="#F8F2ED", zorder=0)
    ax.axvline(len(xs) - 1 - 0.5, color="#D8CFC7", linewidth=1.0, linestyle="--", zorder=1)

    ax.bar([x - metric_width for x in xs], f1_all, width=metric_width, color=COLOR_F1, label="F1", zorder=3)
    ax.bar(xs, acc_all, width=metric_width, color=COLOR_ACC, label="Accuracy", zorder=3)
    ax.bar([x + metric_width for x in xs], rec_all, width=metric_width, color=COLOR_REC, label="Recall", zorder=3)

    style_axis(ax)
    ax.set_xlim(-0.8, len(xs) - 0.2)
    ax.set_ylim(0.72, 1.02)
    ax.set_ylabel("Classification metric", fontsize=9, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)

    ax2 = ax.twinx()
    ax2.plot(xs, runtime_all, color=COLOR_TIME, linewidth=2.0, marker="o", markersize=3.1, label="Runtime", zorder=4)
    ax2.set_ylim(0, 95)
    ax2.set_ylabel("Inference time (min)", fontsize=9, color=MUTED)
    ax2.tick_params(labelsize=8, colors=MUTED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(SPINE)

    best_f1_idx = max(range(len(f1s)), key=lambda i: f1s[i])
    ax.scatter([xs[best_f1_idx] - metric_width], [f1_all[best_f1_idx]], color=COLOR_F1, s=18, zorder=5)

    metric_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_F1),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ACC),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_REC),
        Line2D([0], [0], color=COLOR_TIME, marker="o", linewidth=2.0, markersize=3.2),
    ]
    metric_labels = ["F1", "Accuracy", "Recall", "Runtime"]
    ax.legend(metric_handles, metric_labels, frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 1.01))


    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)



if __name__ == "__main__":
    main()
