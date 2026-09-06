#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROW_SPECS = [
    ("HK97-like", "core_HK97_like.csv"),
    ("picorna-like", "core_Pico_like.csv"),
    ("NCLDV-like", "core_NCLDV-like.csv"),
    ("BTV-like", "core_BTV-like.csv"),
    ("micro-like", "core_Micro-like.csv"),
]


def sanitize_ngram(label: str) -> str:
    return " ".join(str(label).split())


def family_key(ngram: str) -> str:
    tokens = sanitize_ngram(ngram).split()
    if not tokens:
        return ngram
    if len(set(tokens)) == 1:
        return tokens[0]
    return " ".join(tokens)


def display_label_for_family(family: str, members: Iterable[str]) -> str:
    members = [sanitize_ngram(m) for m in members]
    tokens = family.split()
    if len(tokens) == 1:
        max_len = max(len(m.split()) for m in members)
        if max_len > 1:
            return f"{family}x1-{max_len}"
    return family


def load_top_motifs_from_source_dir(
    source_dir: Path,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]], dict[str, str]]:
    selected_by_row: dict[str, list[str]] = {}
    score_maps: dict[str, dict[str, float]] = {}
    ordered_columns: list[str] = []
    column_owner: dict[str, str] = {}
    family_members: dict[str, list[str]] = {}
    display_labels: dict[str, str] = {}
    seen: set[str] = set()

    for row_label, filename in ROW_SPECS:
        path = source_dir / filename
        df = pd.read_csv(path)
        df["ngram"] = df["ngram"].map(sanitize_ngram)
        df = df.sort_values(["mean_tfidf", "ngram"], ascending=[False, True]).reset_index(drop=True)
        top = df.head(top_k).copy()
        top["family"] = top["ngram"].map(family_key)
        grouped = (
            top.groupby("family", as_index=False)
            .agg(mean_tfidf=("mean_tfidf", "max"), members=("ngram", lambda s: list(dict.fromkeys(s.tolist()))))
            .sort_values(["mean_tfidf", "family"], ascending=[False, True])
            .reset_index(drop=True)
        )
        families = grouped["family"].tolist()
        selected_by_row[row_label] = families
        score_maps[row_label] = dict(zip(grouped["family"], grouped["mean_tfidf"]))
        for family, members in zip(grouped["family"], grouped["members"]):
            family_members.setdefault(family, list(dict.fromkeys(members)))
            display_labels[family] = display_label_for_family(family, family_members[family])
            if family not in seen:
                ordered_columns.append(family)
                column_owner[family] = row_label
                seen.add(family)

    matrix = pd.DataFrame(index=[r for r, _ in ROW_SPECS], columns=ordered_columns, dtype=float)
    for row_label, _filename in ROW_SPECS:
        row_scores = score_maps[row_label]
        for family in ordered_columns:
            if family in row_scores:
                matrix.loc[row_label, family] = row_scores[family]

    matrix = matrix.loc[~matrix.isna().all(axis=1)].copy()
    return matrix, column_owner, family_members, display_labels


def build_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "motif_warm",
        [
            "#fbf7ef",
            "#f6dcc1",
            "#eda56f",
            "#d9653b",
            "#8e1f1f",
        ],
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    figures_dir = root / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)

    matrix, column_owner, _family_members, display_labels = load_top_motifs_from_source_dir(data_dir, top_k=8)

    cmap = build_colormap()
    vmax = float(matrix.max().max())
    fig_width = max(14, 3.8 + 0.48 * len(matrix.columns))
    fig_height = 5.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220)
    ax.set_facecolor("#ffffff")

    cmap.set_bad(color="#ffffff")
    image = ax.imshow(
        matrix.to_numpy(),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.55, pad=0.02)
    colorbar.set_label("Mean TF-IDF")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticks([index - 0.5 for index in range(len(matrix.columns) + 1)], minor=True)
    ax.set_yticks([index - 0.5 for index in range(len(matrix.index) + 1)], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([display_labels.get(col, col) for col in matrix.columns], rotation=90, ha="center", va="top", fontsize=8)
    ax.set_yticklabels(matrix.index, rotation=0, fontsize=11)

    owner_breaks = []
    previous_owner = None
    for idx, motif in enumerate(matrix.columns):
        owner = column_owner.get(motif)
        if previous_owner is None:
            previous_owner = owner
            continue
        if owner != previous_owner:
            owner_breaks.append(idx)
        previous_owner = owner
    for xpos in owner_breaks:
        ax.vlines(xpos - 0.5, *ax.get_ylim(), colors="#ddd7cc", linewidth=1.5, zorder=5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(rect=[0.03, 0.06, 0.99, 0.90])
    out_png = figures_dir / "figure_s4.png"
    out_svg = figures_dir / "figure_s4.svg"
    fig.savefig(out_png, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)

if __name__ == "__main__":
    main()
