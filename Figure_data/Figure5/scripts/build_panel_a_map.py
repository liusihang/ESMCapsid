#!/usr/bin/env python3
"""
Build ViCapsid Figure 5A from sequence coordinates and ecosystem assignments.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CATEGORY_ORDER = [
    "General Environmental",
    "Host Associated",
    "Extreme Environments",
    "Engineered Systems",
    "Contaminated and Industrial Environments",
    "Mixed And Lab",
]

CATEGORY_COLORS = {
    "General Environmental": "#f77f5f",
    "Host Associated": "#f8b74a",
    "Extreme Environments": "#d7d95a",
    "Engineered Systems": "#32a89d",
    "Contaminated and Industrial Environments": "#4f97de",
    "Mixed And Lab": "#d96b9a",
}

MAP_PIE_RADIUS_SCALE = 1.35
MAP_PIE_AREA_SCALE = MAP_PIE_RADIUS_SCALE**2

SIZE_BIN_STYLES = [
    {"min": 1, "max": 99, "label": "[1, 100)", "size": 36},
    {"min": 100, "max": 999, "label": "[100, 1,000)", "size": 90},
    {"min": 1000, "max": 4999, "label": "[1,000, 5,000)", "size": 220},
    {"min": 5000, "max": 19999, "label": "[5,000, 20,000)", "size": 520},
    {"min": 20000, "max": None, "label": "[20,000, )", "size": 1100},
]


def normalize(value: object) -> str:
    return "" if value is None else str(value).strip()


def map_ecosystem_category(row: dict[str, str]) -> str:
    main_ecosystem = normalize(row.get("main_ecosystem"))
    return main_ecosystem


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bucket_coordinate(value: float, bin_size: float) -> float:
    return round(value / bin_size) * bin_size


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def aggregate_location_ecosystem_counts(
    coord_rows: list[dict[str, str]],
    ecosystem_rows: list[dict[str, str]],
    bin_size: float = 10.0,
) -> list[dict[str, int | float]]:
    ecosystem_by_sequence = {
        normalize(row.get("sequence_id")): row
        for row in ecosystem_rows
        if normalize(row.get("sequence_id"))
    }
    buckets: dict[tuple[float, float], dict[str, int | float]] = {}

    for coord_row in coord_rows:
        sequence_id = normalize(coord_row.get("sequence_id"))
        latitude_text = normalize(coord_row.get("latitude"))
        longitude_text = normalize(coord_row.get("longitude"))
        if not sequence_id or not latitude_text or not longitude_text:
            continue
        ecosystem_row = ecosystem_by_sequence.get(sequence_id)
        if ecosystem_row is None:
            continue

        latitude = clamp(bucket_coordinate(float(latitude_text), bin_size), -90.0, 90.0)
        longitude = clamp(bucket_coordinate(float(longitude_text), bin_size), -180.0, 180.0)
        key = (latitude, longitude)
        if key not in buckets:
            buckets[key] = {
                "latitude": latitude,
                "longitude": longitude,
                "plot_latitude": 0.0,
                "plot_longitude": 0.0,
                "total_count": 0,
                **{category: 0 for category in CATEGORY_ORDER},
            }

        category = map_ecosystem_category(ecosystem_row)
        if category not in CATEGORY_ORDER:
            continue

        buckets[key]["plot_latitude"] += float(latitude_text)
        buckets[key]["plot_longitude"] += float(longitude_text)
        buckets[key]["total_count"] += 1
        buckets[key][category] += 1

    rows = []
    for key in sorted(buckets):
        row = buckets[key]
        total_count = int(row["total_count"])
        row["plot_latitude"] = float(row["plot_latitude"]) / total_count
        row["plot_longitude"] = float(row["plot_longitude"]) / total_count
        rows.append(row)
    return rows


def export_compiled_table(rows: list[dict[str, int | float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "latitude",
        "longitude",
        "plot_latitude",
        "plot_longitude",
        "total_count",
        "size_bin_label",
        *CATEGORY_ORDER,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["size_bin_label"] = get_size_bin_style(int(row["total_count"]))["label"]
            writer.writerow(enriched)


def get_size_bin_style(total_count: int) -> dict[str, int | str | None]:
    for style in SIZE_BIN_STYLES:
        upper = style["max"]
        if upper is None and total_count >= int(style["min"]):
            return style
        if upper is not None and int(style["min"]) <= total_count <= int(upper):
            return style
    return SIZE_BIN_STYLES[0]


def get_marker_size(total_count: int, marker_size_scale: float = 1.0) -> float:
    base_size = int(get_size_bin_style(total_count)["size"])
    return max(16.0, base_size * marker_size_scale)


def load_compiled_rows(path: str | Path) -> list[dict[str, str]]:
    return load_csv_rows(path)


def make_wedge_marker(start_angle: float, end_angle: float):
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "numpy is required for plotting. Run with `uv run --with numpy --with matplotlib --with cartopy ...`."
        ) from exc

    theta = np.linspace(math.radians(start_angle), math.radians(end_angle), 36)
    return np.vstack(
        [
            [0.0, 0.0],
            np.column_stack([np.cos(theta), np.sin(theta)]),
            [0.0, 0.0],
        ]
    )


def _import_plotting():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib and cartopy are required. Run with `uv run --with numpy --with matplotlib --with cartopy ...`."
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt, Patch, ccrs, cfeature


def _relax_marker_positions(
    *,
    ax,
    ccrs,
    longitudes: list[float],
    latitudes: list[float],
    marker_sizes: list[float],
    iterations: int = 240,
    overlap_scale: float = 0.95,
    min_gap_px: float = 4.0,
    max_shift_px: float = 88.0,
) -> list[tuple[float, float]]:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return list(zip(longitudes, latitudes))

    if not longitudes:
        return []

    fig = ax.figure
    fig.canvas.draw()

    lon_values = np.array(longitudes, dtype=float)
    lat_values = np.array(latitudes, dtype=float)
    projected = ax.projection.transform_points(ccrs.PlateCarree(), lon_values, lat_values)[:, :2]
    display_positions = ax.transData.transform(projected)
    original_positions = display_positions.copy()
    radii_px = np.sqrt(np.array(marker_sizes, dtype=float) / math.pi) * fig.dpi / 72.0

    for _iteration in range(iterations):
        total_motion = 0.0
        for left in range(len(display_positions)):
            for right in range(left + 1, len(display_positions)):
                delta = display_positions[right] - display_positions[left]
                distance = float(np.hypot(delta[0], delta[1]))
                target_distance = (radii_px[left] + radii_px[right]) * overlap_scale + min_gap_px
                if distance >= target_distance:
                    continue

                if distance < 1e-6:
                    angle = ((left * 37 + right * 17) % 360) * math.pi / 180.0
                    direction = np.array([math.cos(angle), math.sin(angle)])
                    distance = 1.0
                else:
                    direction = delta / distance

                push = (target_distance - distance) * 0.5
                total_radius = radii_px[left] + radii_px[right]
                left_weight = radii_px[right] / total_radius if total_radius else 0.5
                right_weight = radii_px[left] / total_radius if total_radius else 0.5

                display_positions[left] -= direction * push * left_weight
                display_positions[right] += direction * push * right_weight
                total_motion += push

        shift = display_positions - original_positions
        shift_lengths = np.hypot(shift[:, 0], shift[:, 1])
        too_far = shift_lengths > max_shift_px
        if np.any(too_far):
            display_positions[too_far] = (
                original_positions[too_far]
                + shift[too_far] * (max_shift_px / shift_lengths[too_far])[:, None]
            )

        if total_motion < 0.05:
            break

    relaxed_projected = ax.transData.inverted().transform(display_positions)
    relaxed_lonlat = ccrs.PlateCarree().transform_points(
        ax.projection,
        relaxed_projected[:, 0],
        relaxed_projected[:, 1],
    )[:, :2]

    results = []
    for index, (longitude, latitude) in enumerate(relaxed_lonlat):
        if np.isfinite(longitude) and np.isfinite(latitude):
            results.append((float(longitude), float(latitude)))
        else:
            results.append((float(longitudes[index]), float(latitudes[index])))
    return results


def draw_world_map(
    *,
    ax,
    compiled_rows: list[dict[str, str | int | float]],
    legend_category_anchor: tuple[float, float] = (0.02, 0.02),
    legend_size_anchor: tuple[float, float] = (0.985, 0.04),
    legend_category_fontsize: float = 11,
    legend_size_fontsize: float = 10.5,
    legend_size_loc: str = "lower right",
    legend_category_loc: str = "lower left",
    legend_category_ncol: int = 3,
    legend_size_ncol: int = 5,
    marker_size_scale: float = 0.72 * MAP_PIE_AREA_SCALE,
    relax_markers: bool = True,
    relax_overlap_scale: float = 0.95,
    relax_max_shift_px: float = 88.0,
) -> None:
    _plt, Patch, ccrs, cfeature = _import_plotting()

    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#ffffff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e9eef6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.55, edgecolor="#4d5766", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.20, edgecolor="#c6cfda", zorder=1)
    ax.set_extent([-180, 180, -82, 86], crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.set_yticks([])

    marker_rows = []
    for row in compiled_rows:
        total_count = int(row["total_count"])
        if total_count <= 0:
            continue
        marker_rows.append(
            {
                "row": row,
                "marker_size": get_marker_size(total_count, marker_size_scale),
                "longitude": float(row["plot_longitude"]),
                "latitude": float(row["plot_latitude"]),
            }
        )

    if relax_markers:
        marker_positions = _relax_marker_positions(
            ax=ax,
            ccrs=ccrs,
            longitudes=[float(item["longitude"]) for item in marker_rows],
            latitudes=[float(item["latitude"]) for item in marker_rows],
            marker_sizes=[float(item["marker_size"]) for item in marker_rows],
            overlap_scale=relax_overlap_scale,
            max_shift_px=relax_max_shift_px,
        )
        marker_transform = ccrs.PlateCarree()
    else:
        marker_positions = [
            (float(item["longitude"]), float(item["latitude"]))
            for item in marker_rows
        ]
        marker_transform = ccrs.PlateCarree()

    for marker_row, (plot_x, plot_y) in zip(marker_rows, marker_positions):
        row = marker_row["row"]
        total_count = int(row["total_count"])
        marker_size = float(marker_row["marker_size"])
        start_angle = 0.0

        for category in CATEGORY_ORDER:
            category_count = int(row[category])
            if category_count <= 0:
                continue
            angle = 360.0 * (category_count / total_count)
            marker = make_wedge_marker(start_angle, start_angle + angle)
            ax.scatter(
                [plot_x],
                [plot_y],
                s=marker_size,
                marker=marker,
                color=CATEGORY_COLORS[category],
                edgecolors="#5d6775",
                linewidths=0.15,
                transform=marker_transform,
                zorder=3,
            )
            start_angle += angle

        ax.scatter(
            [plot_x],
            [plot_y],
            s=marker_size,
            marker="o",
            facecolors="none",
            edgecolors="#5d6775",
            linewidths=0.35,
            transform=marker_transform,
            zorder=4,
        )

    category_handles = [
        Patch(facecolor=CATEGORY_COLORS[category], edgecolor="none", label=category)
        for category in CATEGORY_ORDER
    ]
    legend_categories = fig.legend(
        handles=category_handles,
        loc=legend_category_loc,
        bbox_to_anchor=legend_category_anchor,
        ncol=legend_category_ncol,
        frameon=False,
        fontsize=legend_category_fontsize,
        handlelength=1.2,
        handleheight=1.2,
        columnspacing=1.6,
        handletextpad=0.45,
    )
    fig.add_artist(legend_categories)

    size_handles = []
    for style in SIZE_BIN_STYLES:
        size_handles.append(
            ax.scatter(
                [],
                [],
                s=max(16.0, int(style["size"]) * marker_size_scale),
                marker="o",
                facecolors="#cfcfd1",
                edgecolors="none",
                label=str(style["label"]),
            )
        )
    fig.legend(
        handles=size_handles,
        loc=legend_size_loc,
        bbox_to_anchor=legend_size_anchor,
        ncol=legend_size_ncol,
        frameon=False,
        fontsize=legend_size_fontsize,
        handletextpad=0.6,
        columnspacing=1.25,
    )


def plot_world_map(compiled_rows: list[dict[str, str | int | float]], output_path: str | Path) -> None:
    plt, _Patch, ccrs, _cfeature = _import_plotting()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8.8), dpi=240)
    ax = plt.axes(projection=ccrs.Robinson())
    draw_world_map(ax=ax, compiled_rows=compiled_rows)

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.20)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def get_default_paths(script_path: str | Path) -> dict[str, Path]:
    package_root = Path(script_path).resolve().parent.parent
    return {
        "coordinates": package_root / "data" / "source_data" / "Seqs2Coordinates.csv",
        "ecosystem": package_root / "data" / "source_data" / "Seqs2Ecosystem.csv",
        "compiled": package_root / "data" / "panel_a_location_ecosystem_composition.csv",
        "plot": package_root / "plots" / "panel_a_map_reproduced.png",
    }


def parse_args() -> argparse.Namespace:
    defaults = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", default=str(defaults["coordinates"]))
    parser.add_argument("--ecosystem", default=str(defaults["ecosystem"]))
    parser.add_argument("--compiled-output", default=str(defaults["compiled"]))
    parser.add_argument("--plot-output", default=str(defaults["plot"]))
    parser.add_argument("--bin-size", type=float, default=25.0)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coord_rows = load_csv_rows(args.coordinates)
    ecosystem_rows = load_csv_rows(args.ecosystem)
    compiled_rows = aggregate_location_ecosystem_counts(
        coord_rows=coord_rows,
        ecosystem_rows=ecosystem_rows,
        bin_size=args.bin_size,
    )
    export_compiled_table(compiled_rows, args.compiled_output)
    if not args.build_only:
        plot_world_map(compiled_rows, args.plot_output)


if __name__ == "__main__":
    main()
