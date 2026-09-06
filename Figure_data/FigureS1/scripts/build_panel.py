from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


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


RNG = random.Random(17)

COLORS = {
    "Capsid": "#d1495b",
    "Other": "#2e86ab",
}

OPACITY = {
    "Capsid": 0.55,
    "Other": 0.35,
}

KEEP_LIMITS = {
    "Capsid": None,
    "Other": 12000,
}

MODEL_ORDER = [
    ("esmc_600m", "ESMC 600M"),
    ("esm3o", "ESM3O"),
    ("esm2_650m", "ESM2 650M"),
    ("profluent_e1_600m", "Profluent E1 600M"),
]


def reservoir_sample(rows: list[tuple[float, float]], limit: int | None) -> list[tuple[float, float]]:
    if limit is None or len(rows) <= limit:
        return rows
    sample = rows[:limit]
    for idx in range(limit, len(rows)):
        j = RNG.randint(0, idx)
        if j < limit:
            sample[j] = rows[idx]
    return sample


def load_points(path: Path) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = {"Capsid": [], "Other": []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row["label"])
            if label in grouped:
                grouped[label].append((float(row["dim1"]), float(row["dim2"])))
    return {label: reservoir_sample(rows, KEEP_LIMITS[label]) for label, rows in grouped.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render one Figure S1 model panel.")
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", choices=[key for key, _ in MODEL_ORDER], required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_key = args.model
    display_name = dict(MODEL_ORDER)[model_key]
    coord_csv = results_root / model_key / "coords.csv"
    grouped = load_points(coord_csv)
    output_stem = output_dir / f"panel_{model_key}_reproduced"

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=240)
    for label in ["Other", "Capsid"]:
        rows = grouped[label]
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        ax.scatter(
            xs,
            ys,
            s=3 if label == "Other" else 5,
            c=COLORS[label],
            alpha=OPACITY[label],
            linewidths=0,
            label=label,
            rasterized=True,
        )

    ax.set_title(display_name, fontsize=12)
    ax.set_xlabel("t-SNE dim1")
    ax.set_ylabel("t-SNE dim2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
