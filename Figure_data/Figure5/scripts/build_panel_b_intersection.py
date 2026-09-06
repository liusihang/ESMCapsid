#!/usr/bin/env python3
"""
Build ViCapsid Figure 5B from the packaged ecosystem-sharing tables.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


MAIN_ECOSYSTEM_ORDER = [
    "General Environmental",
    "Host Associated",
    "Extreme Environments",
    "Engineered Systems",
    "Contaminated and Industrial Environments",
    "Mixed And Lab",
]

SHARING_DEGREES = [6, 5, 4, 3, 2, 1]

SHARING_LABELS = {
    6: "Shared 6",
    5: "Shared 5",
    4: "Shared 4",
    3: "Shared 3",
    2: "Shared 2",
    1: "Unique",
}

ECO_COLORS = {
    "General Environmental": "#f77f5f",
    "Host Associated": "#f8b74a",
    "Extreme Environments": "#d7d95a",
    "Engineered Systems": "#32a89d",
    "Contaminated and Industrial Environments": "#4f97de",
    "Mixed And Lab": "#d96b9a",
}

MAP_CATEGORY_ORDER = [
    "Aquatic",
    "Terrestrial",
    "Extreme",
    "Host",
    "Engineered",
    "Anthropogenic",
]

MAP_CATEGORY_COLORS = {
    "Aquatic": "#f77f5f",
    "Terrestrial": "#f8b74a",
    "Extreme": "#d7d95a",
    "Host": "#32a89d",
    "Engineered": "#4f97de",
    "Anthropogenic": "#d96b9a",
}

ECO_ABBREV = {
    "General Environmental": "GenEnv",
    "Host Associated": "HostAs",
    "Extreme Environments": "ExtEnv",
    "Engineered Systems": "EngSys",
    "Contaminated and Industrial Environments": "ContInd",
    "Mixed And Lab": "MixLab",
}

ECO_DISPLAY_LABEL = {
    "General Environmental": "General\nEnvironmental",
    "Host Associated": "Host\nAssociated",
    "Extreme Environments": "Extreme\nEnvironments",
    "Engineered Systems": "Engineered\nSystems",
    "Contaminated and Industrial Environments": "Contaminated and Industrial\nEnvironments",
    "Mixed And Lab": "Mixed\nAnd Lab",
}

COUNT_LOG_LINTHRESH = 1_000
DOT_SIZE_POWER = 0.5
DOT_LEGEND_VALUES = [1_000, 10_000, 100_000]
UNANNOTATED_COLOR = "#d8dde2"


def normalize(value: object) -> str:
    return "" if value is None else str(value).strip()


def get_default_hmm_annotation_path(path: str | Path) -> Path | None:
    candidate = Path(path).resolve().parent / "panel_b_hmm_annotation.csv"
    return candidate if candidate.exists() else None


def load_hmm_annotation_data(path: str | Path | None) -> dict[tuple[str, str], dict[str, float | int]]:
    if path is None:
        return {}

    required_fields = {
        "main_ecosystem",
        "sub_ecosystem",
        "hmm_hit_count",
        "total_sequence_count",
        "hmm_hit_rate",
    }
    annotations: dict[tuple[str, str], dict[str, float | int]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                "Figure 5B HMM annotation table is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        for row in reader:
            main_ecosystem = normalize(row.get("main_ecosystem"))
            sub_ecosystem = normalize(row.get("sub_ecosystem"))
            if not main_ecosystem or not sub_ecosystem:
                continue
            annotations[(main_ecosystem, sub_ecosystem)] = {
                "hmm_hit_count": int(normalize(row.get("hmm_hit_count")) or 0),
                "hmm_total_sequence_count": int(normalize(row.get("total_sequence_count")) or 0),
                "hmm_hit_rate": float(normalize(row.get("hmm_hit_rate")) or 0.0),
            }
    return annotations


def load_upset_data(
    path: str | Path,
    *,
    hmm_annotation_path: str | Path | None = None,
) -> dict[str, object]:
    required_fields = {
        "main_ecosystem",
        "sub_ecosystem",
        "sharing_degree",
        "sequence_count",
        "cluster_count",
        "sub_ecosystem_total_sequence_count",
    }
    cell_sequence_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    cell_cluster_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    column_totals: dict[tuple[str, str], int] = {}
    present_main_ecosystems: set[str] = set()

    input_path = Path(path)
    if hmm_annotation_path is None:
        hmm_annotation_path = get_default_hmm_annotation_path(input_path)
    hmm_annotations = load_hmm_annotation_data(hmm_annotation_path)

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(
                "Figure 5B subecosystem sharing table is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        for row in reader:
            main_ecosystem = normalize(row.get("main_ecosystem"))
            sub_ecosystem = normalize(row.get("sub_ecosystem"))
            if not main_ecosystem or not sub_ecosystem:
                continue
            sharing_degree = int(normalize(row.get("sharing_degree")) or 0)
            if sharing_degree not in SHARING_DEGREES:
                continue
            sequence_count = int(normalize(row.get("sequence_count")) or 0)
            cluster_count = int(normalize(row.get("cluster_count")) or 0)
            total_sequence_count = int(
                normalize(row.get("sub_ecosystem_total_sequence_count")) or 0
            )
            key = (main_ecosystem, sub_ecosystem, sharing_degree)
            cell_sequence_counts[key] += sequence_count
            cell_cluster_counts[key] += cluster_count
            column_totals[(main_ecosystem, sub_ecosystem)] = total_sequence_count
            present_main_ecosystems.add(main_ecosystem)

    ordered_main_ecosystems = [
        ecosystem for ecosystem in MAIN_ECOSYSTEM_ORDER if ecosystem in present_main_ecosystems
    ]
    ordered_main_ecosystems.extend(
        sorted(present_main_ecosystems.difference(MAIN_ECOSYSTEM_ORDER))
    )

    columns: list[dict[str, object]] = []
    for main_ecosystem in ordered_main_ecosystems:
        pairs = [
            (sub_ecosystem, total)
            for (main, sub_ecosystem), total in column_totals.items()
            if main == main_ecosystem
        ]
        for sub_ecosystem, total in sorted(pairs, key=lambda item: (-item[1], item[0])):
            annotation = hmm_annotations.get((main_ecosystem, sub_ecosystem))
            hmm_hit_count = None
            hmm_unannotated_count = None
            hmm_hit_rate = None
            if annotation is not None:
                hmm_hit_count = min(int(annotation["hmm_hit_count"]), total)
                hmm_unannotated_count = max(total - hmm_hit_count, 0)
                hmm_hit_rate = (hmm_hit_count / total) if total else 0.0
            columns.append(
                {
                    "main_ecosystem": main_ecosystem,
                    "sub_ecosystem": sub_ecosystem,
                    "total_sequence_count": total,
                    "hmm_hit_count": hmm_hit_count,
                    "hmm_unannotated_count": hmm_unannotated_count,
                    "hmm_hit_rate": hmm_hit_rate,
                }
            )

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "numpy is required to load Figure 5B subecosystem sharing data."
        ) from exc
    sequence_count_matrix = np.zeros((len(SHARING_DEGREES), len(columns)), dtype=int)
    cluster_count_matrix = np.zeros((len(SHARING_DEGREES), len(columns)), dtype=int)
    degree_to_row = {degree: index for index, degree in enumerate(SHARING_DEGREES)}
    for column_index, column in enumerate(columns):
        main_ecosystem = str(column["main_ecosystem"])
        sub_ecosystem = str(column["sub_ecosystem"])
        for sharing_degree in SHARING_DEGREES:
            row_index = degree_to_row[sharing_degree]
            key = (main_ecosystem, sub_ecosystem, sharing_degree)
            sequence_count_matrix[row_index, column_index] = cell_sequence_counts.get(key, 0)
            cluster_count_matrix[row_index, column_index] = cell_cluster_counts.get(key, 0)

    return {
        "columns": columns,
        "main_ecosystems": ordered_main_ecosystems,
        "sharing_degrees": list(SHARING_DEGREES),
        "sharing_labels": [SHARING_LABELS[degree] for degree in SHARING_DEGREES],
        "sequence_count_matrix": sequence_count_matrix,
        "cluster_count_matrix": cluster_count_matrix,
    }


def get_default_paths(script_path: str | Path) -> dict[str, Path]:
    package_root = Path(script_path).resolve().parent.parent
    return {
        "input": package_root / "data" / "panel_b_cluster_sharing.csv",
        "hmm_annotation": package_root / "data" / "panel_b_hmm_annotation.csv",
        "plot": package_root / "plots" / "panel_b_intersection_reproduced.png",
    }


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.ticker import FuncFormatter
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib and numpy are required to render Figure 5B."
        ) from exc
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt, mpatches, FuncFormatter, np


def format_compact_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_compact_axis_count(value: float) -> str:
    if abs(value) < 1:
        return "0"
    return format_compact_count(int(round(value)))


def get_positive_log_ticks(max_value: int) -> list[int]:
    if max_value <= 0:
        return [1]
    ticks = []
    value = 100
    upper = max(100, int(max_value))
    while value <= upper:
        ticks.append(value)
        value *= 10
    return ticks or [100]


def map_column_category(main_ecosystem: str, sub_ecosystem: str) -> str:
    if sub_ecosystem.startswith("Aquatic"):
        return "Aquatic"
    if sub_ecosystem in {"Plants", "Terrestrial Soil", "Fungi And Algae"}:
        return "Terrestrial"
    if main_ecosystem == "Extreme Environments" or sub_ecosystem in {
        "Psychrophilic Low Temperature",
        "Thermophilic High Temperature",
        "Subsurface And Geologic",
    }:
        return "Extreme"
    if main_ecosystem == "Engineered Systems" or sub_ecosystem in {
        "Bioreactors",
        "Wastewater Treatment",
        "Industrial Wastewater",
        "Built Environment",
        "Solid Waste Management",
        "Controlled Environment",
    }:
        return "Engineered"
    if main_ecosystem in {"Contaminated and Industrial Environments", "Mixed And Lab"} or sub_ecosystem in {
        "Chemical Remediation Targets",
        "Landfill",
        "Lab Enrichment",
        "Unknown Mixed",
    }:
        return "Anthropogenic"
    if main_ecosystem == "Host Associated" or sub_ecosystem in {"Human", "Animals", "Microbial Hosts"}:
        return "Host"
    return "Terrestrial"


def get_column_color(main_ecosystem: str, sub_ecosystem: str) -> str:
    return ECO_COLORS.get(main_ecosystem, "#7f7f7f")


def lighten_hex_color(color: str, white_fraction: float = 0.70) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "#d8dde2"
    channels = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
    lightened = [
        int(channel * (1.0 - white_fraction) + 255 * white_fraction)
        for channel in channels
    ]
    return "#" + "".join(f"{channel:02x}" for channel in lightened)


def get_annotated_bar_color(main_ecosystem: str, sub_ecosystem: str) -> str:
    return lighten_hex_color(get_column_color(main_ecosystem, sub_ecosystem))


def get_hmm_bar_segments(column: dict[str, object]) -> tuple[dict[str, int | str], dict[str, int | str]]:
    hit_count = int(column["hmm_hit_count"]) if column.get("hmm_hit_count") is not None else int(column["total_sequence_count"])
    unannotated_count = (
        int(column["hmm_unannotated_count"])
        if column.get("hmm_unannotated_count") is not None
        else 0
    )
    bottom = {
        "label": "Not annotated",
        "height": unannotated_count,
        "bottom": 0,
    }
    top = {
        "label": "HMM annotated",
        "height": hit_count,
        "bottom": unannotated_count,
    }
    return bottom, top


def scale_sequence_dot_size(
    value: int,
    *,
    min_value: int,
    max_value: int,
    min_size: float = 6.0,
    max_size: float = 260.0,
    power: float = DOT_SIZE_POWER,
) -> float:
    if value <= 0:
        return 0.0
    if max_value <= min_value:
        return (min_size + max_size) / 2
    safe_min = max(0, int(min_value))
    safe_max = max(safe_min + 1, int(max_value))
    fraction = (int(value) - safe_min) / (safe_max - safe_min)
    fraction = max(0.0, min(1.0, fraction))
    fraction = fraction**power
    return min_size + (max_size - min_size) * fraction


def scale_sequence_dot_sizes(sequence_counts: list[int]) -> list[float]:
    positive_counts = [max(1, int(value)) for value in sequence_counts if int(value) > 0]
    if not positive_counts:
        return []
    min_count = min(positive_counts)
    max_count = max(positive_counts)
    return [
        scale_sequence_dot_size(value, min_value=min_count, max_value=max_count)
        for value in sequence_counts
    ]


def get_dot_legend_values(max_value: int) -> list[int]:
    return [value for value in DOT_LEGEND_VALUES if value <= max_value]


def draw_sequence_dot_size_legend(ax, sequence_counts: list[int]) -> None:
    _plt, _mpatches, _FuncFormatter, np = _import_plotting()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    positive_counts = [int(value) for value in sequence_counts if int(value) > 0]
    if not positive_counts:
        return

    min_count = min(positive_counts)
    max_count = max(positive_counts)
    unique_values = get_dot_legend_values(max_count)
    if not unique_values:
        unique_values = [max_count]

    ax.text(
        0.08,
        0.94,
        "Sequences\n(sqrt dot area)",
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
        color="#333333",
    )
    y_values = np.linspace(0.62, 0.20, len(unique_values))
    for y_value, count in zip(y_values, unique_values):
        dot_size = scale_sequence_dot_size(count, min_value=min_count, max_value=max_count)
        ax.scatter(
            [0.34],
            [float(y_value)],
            s=dot_size,
            c="#666666",
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
        ax.text(
            0.50,
            float(y_value),
            format_compact_count(count),
            ha="left",
            va="center",
            fontsize=6.5,
            color="#333333",
        )


def scale_cluster_dot_size(*args, **kwargs) -> float:
    return scale_sequence_dot_size(*args, **kwargs)


def scale_cluster_dot_sizes(cluster_counts: list[int]) -> list[float]:
    return scale_sequence_dot_sizes(cluster_counts)


def get_sharing_degree_sequence_totals(sequence_count_matrix) -> list[int]:
    return [int(value) for value in sequence_count_matrix.sum(axis=1).tolist()]


def get_sharing_degree_cluster_totals(cluster_count_matrix) -> list[int]:
    return [int(value) for value in cluster_count_matrix.sum(axis=1).tolist()]


def abbreviate_subecosystem_label(label: str) -> str:
    replacements = {
        "Aquatic Freshwater": "Freshwater",
        "Aquatic Marine": "Marine",
        "Aquatic Special": "Aquatic Special",
        "Terrestrial Vegetation": "Vegetation",
        "Terrestrial Soil": "Soil",
        "Psychrophilic Low Temperature": "Psychrophilic",
        "Thermophilic High Temperature": "Thermophilic",
        "Subsurface And Geologic": "Subsurface",
        "Halophilic And Alkaliphilic": "Halophilic",
        "Wastewater Treatment": "Wastewater",
        "Built Environment": "Built Env.",
        "Controlled Environment": "Controlled Env.",
        "Solid Waste Management": "Solid Waste",
        "Food Production": "Food Prod.",
        "Industrial Processing": "Ind. Processing",
        "Contaminated and Industrial Environments": "Anthropogenic",
        "Chemical Remediation Targets": "Chem. Remediation",
        "Industrial Wastewater": "Ind. Wastewater",
        "Hydrocarbon Contamination Oil": "Oil Contam.",
        "Industrial Hydrocarbon Processing": "Hydrocarbon Proc.",
        "Unknown Mixed": "Unknown Mixed",
        "Lab Enrichment": "Lab Enrich.",
        "Microbial Hosts": "Microbial Hosts",
        "Fungi And Algae": "Fungi/Algae",
    }
    if label in replacements:
        return replacements[label]
    if len(label) <= 18:
        return label
    return label[:16].rstrip() + "."


def get_column_group_ranges(columns: list[dict[str, object]]) -> list[dict[str, object]]:
    ranges: list[dict[str, object]] = []
    if not columns:
        return ranges

    start = 0
    current = str(columns[0]["main_ecosystem"])
    for index, column in enumerate(columns[1:], start=1):
        main_ecosystem = str(column["main_ecosystem"])
        if main_ecosystem == current:
            continue
        ranges.append({"main_ecosystem": current, "start": start, "end": index - 1})
        start = index
        current = main_ecosystem
    ranges.append({"main_ecosystem": current, "start": start, "end": len(columns) - 1})
    return ranges


def subset_dataset_columns(dataset: dict[str, object], max_columns: int | None) -> dict[str, object]:
    if max_columns is None:
        return dataset
    max_columns = max(1, int(max_columns))
    columns = list(dataset["columns"])[:max_columns]
    sequence_count_matrix = dataset["sequence_count_matrix"][:, :max_columns]
    cluster_count_matrix = dataset["cluster_count_matrix"][:, :max_columns]
    main_ecosystems = []
    for column in columns:
        main_ecosystem = str(column["main_ecosystem"])
        if main_ecosystem not in main_ecosystems:
            main_ecosystems.append(main_ecosystem)
    return {
        **dataset,
        "columns": columns,
        "sequence_count_matrix": sequence_count_matrix,
        "cluster_count_matrix": cluster_count_matrix,
        "main_ecosystems": main_ecosystems,
    }


def draw_upset_plot(
    dataset: dict[str, object],
    *,
    fig,
    gridspec,
    legend_anchor: tuple[float, float] = (0.56, 1.03),
    show_legend: bool = True,
    max_combos: int | None = None,
) -> None:
    _plt, mpatches, FuncFormatter, np = _import_plotting()

    dataset = subset_dataset_columns(dataset, max_combos)
    columns = list(dataset["columns"])
    sequence_count_matrix = dataset["sequence_count_matrix"]
    cluster_count_matrix = dataset["cluster_count_matrix"]
    sharing_labels = list(dataset["sharing_labels"])
    n_columns = len(columns)
    n_rows = len(sharing_labels)
    ax_matrix = fig.add_subplot(gridspec[1, 2])
    ax_top = fig.add_subplot(gridspec[0, 2], sharex=ax_matrix)
    ax_rowbar = fig.add_subplot(gridspec[1, 0], sharey=ax_matrix)
    ax_left = fig.add_subplot(gridspec[1, 1], sharey=ax_matrix)
    ax_top_label = fig.add_subplot(gridspec[0, 0:2])
    ax_right_blank = fig.add_subplot(gridspec[1, 3])
    ax_right_top = fig.add_subplot(gridspec[0, 3])
    ax_right_blank.axis("off")
    ax_right_top.axis("off")
    all_sequence_counts = [int(value) for value in sequence_count_matrix.flatten() if int(value) > 0]
    draw_sequence_dot_size_legend(ax_top_label, all_sequence_counts)

    x_positions = np.arange(n_columns)
    top_counts = [int(column["total_sequence_count"]) for column in columns]
    bar_colors = [
        get_column_color(str(column["main_ecosystem"]), str(column["sub_ecosystem"]))
        for column in columns
    ]
    annotated_bar_colors = [
        get_annotated_bar_color(str(column["main_ecosystem"]), str(column["sub_ecosystem"]))
        for column in columns
    ]
    has_hmm_annotation = any(column.get("hmm_hit_count") is not None for column in columns)
    hmm_bottom_segments = [get_hmm_bar_segments(column)[0] for column in columns]
    hmm_top_segments = [get_hmm_bar_segments(column)[1] for column in columns]
    group_ranges = get_column_group_ranges(columns)
    for group_index, group in enumerate(group_ranges):
        alpha = 0.055 if group_index % 2 == 0 else 0.025
        ax_top.axvspan(group["start"] - 0.5, group["end"] + 0.5, color="#6f7782", alpha=alpha, zorder=0)
        ax_matrix.axvspan(group["start"] - 0.5, group["end"] + 0.5, color="#6f7782", alpha=alpha, zorder=0)

    ax_top.bar(
        x_positions,
        [int(segment["height"]) for segment in hmm_bottom_segments] if has_hmm_annotation else top_counts,
        width=0.72,
        color=bar_colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.95,
        label="Not annotated",
    )
    if has_hmm_annotation:
        ax_top.bar(
            x_positions,
            [int(segment["height"]) for segment in hmm_top_segments],
            width=0.72,
            bottom=[int(segment["bottom"]) for segment in hmm_top_segments],
            color=annotated_bar_colors,
            edgecolor=bar_colors,
            linewidth=0.25,
            alpha=0.90,
            label="HMM annotated",
        )
    ax_top.set_yscale("symlog", linthresh=COUNT_LOG_LINTHRESH, linscale=0.5, base=10)
    if top_counts:
        ax_top.set_ylim(0, max(top_counts) * 3.2)
        ax_top.set_yticks(get_positive_log_ticks(max(top_counts)))
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: format_compact_axis_count(x)))
    ax_top.set_ylabel("Sequences per\nsub-ecosystem", fontsize=8, fontweight="bold")
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    min_count_to_annotate = 100_000
    for index, count in enumerate(top_counts):
        if count >= min_count_to_annotate:
            label = format_compact_count(count)
            if has_hmm_annotation and columns[index].get("hmm_hit_rate") is not None:
                label = f"{label} / {float(columns[index]['hmm_hit_rate']) * 100:.0f}%"
            ax_top.text(
                index,
                count * 1.10,
                label,
                ha="center",
                va="bottom",
                fontsize=4.5,
                rotation=90,
                color="#333333",
            )

    if all_sequence_counts:
        min_sequence_count = min(all_sequence_counts)
        max_sequence_count = max(all_sequence_counts)
    else:
        min_sequence_count = 1
        max_sequence_count = 1

    for row_index in range(n_rows):
        for column_index, column in enumerate(columns):
            sequence_count = int(sequence_count_matrix[row_index, column_index])
            if sequence_count > 0:
                dot_size = scale_sequence_dot_size(
                    sequence_count,
                    min_value=min_sequence_count,
                    max_value=max_sequence_count,
                )
                ax_matrix.scatter(
                    column_index,
                    row_index,
                    s=dot_size,
                    c=get_column_color(str(column["main_ecosystem"]), str(column["sub_ecosystem"])),
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                    clip_on=False,
                )
            else:
                ax_matrix.scatter(
                    column_index,
                    row_index,
                    s=13,
                    c="white",
                    edgecolors="#d3d3d3",
                    linewidths=0.45,
                    zorder=2,
                    clip_on=False,
                )

    for group in group_ranges[:-1]:
        boundary = group["end"] + 0.5
        ax_top.axvline(boundary, color="#333333", linewidth=0.55, alpha=0.38)
        ax_matrix.axvline(boundary, color="#333333", linewidth=0.55, alpha=0.38)

    for group in group_ranges:
        main_ecosystem = str(group["main_ecosystem"])
        center = (int(group["start"]) + int(group["end"])) / 2
        ax_top.text(
            center,
            1.02,
            ECO_DISPLAY_LABEL.get(main_ecosystem, main_ecosystem),
            transform=ax_top.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#333333",
            clip_on=False,
        )

    ax_matrix.set_xlim(-0.6, n_columns - 0.4)
    ax_matrix.set_ylim(n_rows - 0.5, -0.5)
    ax_matrix.set_yticks(range(n_rows))
    ax_matrix.set_yticklabels([])
    ax_matrix.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels(
        [abbreviate_subecosystem_label(str(column["sub_ecosystem"])) for column in columns],
        fontsize=5.8,
        rotation=65,
        ha="right",
        rotation_mode="anchor",
    )
    ax_matrix.tick_params(axis="x", pad=2)
    ax_matrix.grid(axis="y", color="#d0d0d0", linewidth=0.45, alpha=0.65)
    ax_matrix.set_axisbelow(True)

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(n_rows - 0.5, -0.5)
    ax_left.axis("off")
    for row_index, label in enumerate(sharing_labels):
        ax_left.text(
            0.98,
            row_index,
            label,
            ha="right",
            va="center",
            fontsize=7.6,
            color="#333333",
        )

    row_cluster_totals = get_sharing_degree_cluster_totals(cluster_count_matrix)
    y_positions = np.arange(n_rows)
    ax_rowbar.barh(
        y_positions,
        row_cluster_totals,
        height=0.55,
        color="#5f6368",
        edgecolor="none",
        alpha=0.86,
        zorder=3,
    )
    ax_rowbar.set_ylim(n_rows - 0.5, -0.5)
    ax_rowbar.set_xscale("symlog", linthresh=COUNT_LOG_LINTHRESH, linscale=0.5, base=10)
    if row_cluster_totals:
        max_row_total = max(row_cluster_totals)
        ax_rowbar.set_xlim(max_row_total * 2.7, 0)
        ax_rowbar.set_xticks(get_positive_log_ticks(max_row_total))
    ax_rowbar.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: format_compact_axis_count(x)))
    ax_rowbar.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_rowbar.tick_params(axis="x", labelsize=6.2, pad=1)
    ax_rowbar.grid(axis="x", alpha=0.26, linestyle="--", linewidth=0.45)
    ax_rowbar.set_axisbelow(True)
    ax_rowbar.set_xlabel("Clusters by\nsharing level", fontsize=7, fontweight="bold", labelpad=2)
    for row_index, count in enumerate(row_cluster_totals):
        ax_rowbar.text(
            count * 1.10,
            row_index,
            format_compact_count(count),
            ha="right",
            va="center",
            fontsize=5.8,
            color="#333333",
        )
    for spine in ("top", "right"):
        ax_rowbar.spines[spine].set_visible(False)

    if show_legend:
        legend_handles = [
            mpatches.Patch(color=ECO_COLORS[ecosystem], label=ECO_ABBREV.get(ecosystem, ecosystem))
            for ecosystem in MAIN_ECOSYSTEM_ORDER
            if ecosystem in dataset["main_ecosystems"]
        ]
        if has_hmm_annotation:
            legend_handles.append(
                mpatches.Patch(
                    facecolor=lighten_hex_color(ECO_COLORS["General Environmental"]),
                    edgecolor=ECO_COLORS["General Environmental"],
                    label="Light shade = HMM annotated",
                )
            )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=legend_anchor,
            ncol=4,
            fontsize=6,
            frameon=False,
        )
def plot_upset(dataset: dict[str, object], output_path: str | Path) -> None:
    plt, _mpatches, _FuncFormatter, _np = _import_plotting()

    columns = list(dataset["columns"])
    figure_width = max(13.5, len(columns) * 0.38)
    fig = plt.figure(figsize=(figure_width, 8.2), dpi=300)
    gridspec = fig.add_gridspec(
        2,
        4,
        width_ratios=[2.0, 1.45, 10, 0.18],
        height_ratios=[2.5, 1],
        hspace=0.08,
        wspace=0.04,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.90)

    draw_upset_plot(dataset, fig=fig, gridspec=gridspec)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    defaults = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(defaults["input"]))
    parser.add_argument("--hmm-annotation", default=str(defaults["hmm_annotation"]))
    parser.add_argument("--plot-output", default=str(defaults["plot"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_upset_data(args.input, hmm_annotation_path=args.hmm_annotation)
    plot_upset(dataset, args.plot_output)


if __name__ == "__main__":
    main()
