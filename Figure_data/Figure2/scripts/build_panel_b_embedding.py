from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT_CSV = Path(__file__).resolve().parents[1] / "data/panel_b_embedding_coords.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "plots"
OUT_PNG = OUT_DIR / "panel_b_embedding_reproduced.png"
OUT_SVG = OUT_DIR / "panel_b_embedding_reproduced.svg"

RNG = random.Random(8)

KEEP_LIMITS = {
    "Capsid": None,
    "Other": 14000,
    "Structure": 7000,
    "Cellular": 18000,
}

COLORS = {
    "Capsid": "#C63D2F",
    "Other": "#2C7A7B",
    "Structure": "#D99A2B",
    "Cellular": "#A7B3BE",
}

ALPHAS = {
    "Capsid": 0.90,
    "Other": 0.35,
    "Structure": 0.42,
    "Cellular": 0.16,
}

SIZES = {
    "Capsid": 6.0,
    "Other": 3.2,
    "Structure": 3.4,
    "Cellular": 2.4,
}

PLOT_ORDER = ["Cellular", "Other", "Structure", "Capsid"]
DISPLAY_NAMES = {
    "Capsid": "Capsid",
    "Other": "Other Virus protein",
    "Structure": "Structure protein",
    "Cellular": "Cellular protein",
}


def reservoir_sample(rows: list[tuple[float, float]], limit: int | None) -> list[tuple[float, float]]:
    if limit is None or len(rows) <= limit:
        return rows
    sample = rows[:limit]
    for idx in range(limit, len(rows)):
        j = RNG.randint(0, idx)
        if j < limit:
            sample[j] = rows[idx]
    return sample


def load_points() -> tuple[dict[str, list[tuple[float, float]]], Counter]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    counts: Counter = Counter()
    with INPUT_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["label"]
            point = (float(row["dim1"]), float(row["dim2"]))
            grouped.setdefault(label, []).append(point)
            counts[label] += 1
    return grouped, counts


def make_plot() -> None:
    grouped, counts = load_points()
    sampled = {
        label: reservoir_sample(rows, KEEP_LIMITS.get(label))
        for label, rows in grouped.items()
    }

    fig, ax = plt.subplots(figsize=(7.0, 6.6), dpi=320)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for label in PLOT_ORDER:
        rows = sampled.get(label, [])
        if not rows:
            continue
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        xmin = min(xmin, min(xs))
        xmax = max(xmax, max(xs))
        ymin = min(ymin, min(ys))
        ymax = max(ymax, max(ys))
        ax.scatter(
            xs,
            ys,
            s=SIZES[label],
            c=COLORS[label],
            alpha=ALPHAS[label],
            linewidths=0,
            rasterized=True,
            zorder=5 if label == "Capsid" else 3,
        )

    xpad = (xmax - xmin) * 0.04
    ypad = (ymax - ymin) * 0.04
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = []
    for label in ["Capsid", "Other", "Structure", "Cellular"]:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS[label],
                markeredgewidth=0,
                alpha=min(1.0, ALPHAS[label] + 0.15),
                markersize=6.8 if label == "Capsid" else 5.8,
                label=f"{DISPLAY_NAMES[label]}  ({counts[label]:,})",
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=8.8,
        handletextpad=0.5,
        borderaxespad=0.2,
        labelspacing=0.7,
    )

    fig.tight_layout(pad=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT_SVG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)



if __name__ == "__main__":
    make_plot()
