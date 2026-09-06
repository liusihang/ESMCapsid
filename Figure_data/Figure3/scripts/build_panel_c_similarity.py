#!/usr/bin/env python3
"""Render the final TM-align Figure 3C panel, with optional composite overlay."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "panel_c_similarity"
PLOT_DIR = ROOT / "plots" / "fig3_similarity"
PANEL_WIDTH = 218.0
PANEL_HEIGHT = 176.0
PANEL_X = 30.0
PANEL_Y = 5.0

CATEGORIES = [
    ("within_cluster", "Within\ncluster", "#6FB6C3"),
    ("between_clusters_same_group", "Same\narchitecture", "#D39A58"),
    ("between_groups", "Different\narchitectures", "#7F9FD1"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def category(row: dict[str, str]) -> str:
    if row["same_cluster"].lower() == "true":
        return "within_cluster"
    if row["same_group"].lower() == "true":
        return "between_clusters_same_group"
    return "between_groups"


def y_coord(value: float, bottom: float, height: float) -> float:
    return bottom + value * height


def draw_text_center(c: canvas.Canvas, text: str, x: float, y: float, size: float) -> None:
    c.setFont("Helvetica", size)
    for index, line in enumerate(text.split("\n")):
        c.drawCentredString(x, y - index * (size + 0.8), line)


def render_panel(
    rows: list[dict[str, str]],
    tests: list[dict[str, str]],
    output: Path | None = None,
    *,
    canvas_obj: canvas.Canvas | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
) -> None:
    standalone = canvas_obj is None
    if standalone:
        if output is None:
            raise ValueError("output is required for a standalone panel")
        output.parent.mkdir(parents=True, exist_ok=True)
        c = canvas.Canvas(str(output), pagesize=(PANEL_WIDTH, PANEL_HEIGHT))
    else:
        c = canvas_obj
    c.saveState()
    c.translate(*origin)
    c.setFillColor(white)
    c.rect(0, 0, PANEL_WIDTH, PANEL_HEIGHT, fill=1, stroke=0)

    left, right, bottom, top = 40.0, 7.0, 46.0, 12.0
    width = PANEL_WIDTH - left - right
    height = PANEL_HEIGHT - bottom - top
    x_positions = [left + width * fraction for fraction in (0.18, 0.50, 0.82)]
    values: dict[str, np.ndarray] = {}
    for name, _, _ in CATEGORIES:
        values[name] = np.asarray([float(row["mean_score"]) for row in rows if category(row) == name])

    c.setStrokeColor(HexColor("#D1D5DB"))
    c.setLineWidth(0.35)
    c.setFillColor(HexColor("#6B7280"))
    c.setFont("Helvetica", 5.5)
    for tick in np.linspace(0, 1, 6):
        y = y_coord(float(tick), bottom, height)
        c.line(left, y, left + width, y)
        c.drawRightString(left - 4, y - 1.8, f"{tick:.1f}")

    c.setStrokeColor(HexColor("#64748B"))
    c.setLineWidth(0.6)
    c.line(left, bottom, left, bottom + height)
    c.line(left, bottom, left + width, bottom)

    rng = np.random.default_rng(73)
    box_width = 24.0
    for x, (name, label, color_hex) in zip(x_positions, CATEGORIES, strict=True):
        v = values[name]
        color = HexColor(color_hex)
        q10, q25, q50, q75, q90 = np.quantile(v, [0.10, 0.25, 0.50, 0.75, 0.90])
        c.setStrokeColor(color)
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.15))
        c.setLineWidth(0.7)
        c.line(x, y_coord(float(q10), bottom, height), x, y_coord(float(q25), bottom, height))
        c.line(x, y_coord(float(q75), bottom, height), x, y_coord(float(q90), bottom, height))
        c.line(x - 5, y_coord(float(q10), bottom, height), x + 5, y_coord(float(q10), bottom, height))
        c.line(x - 5, y_coord(float(q90), bottom, height), x + 5, y_coord(float(q90), bottom, height))
        c.rect(x - box_width / 2, y_coord(float(q25), bottom, height), box_width, y_coord(float(q75), bottom, height) - y_coord(float(q25), bottom, height), fill=1, stroke=1)
        c.setStrokeColor(color)
        c.line(x - box_width / 2, y_coord(float(q50), bottom, height), x + box_width / 2, y_coord(float(q50), bottom, height))
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.32))
        for value, dx in zip(v, rng.uniform(-10, 10, size=v.size), strict=True):
            c.circle(x + float(dx), y_coord(float(value), bottom, height), 0.9, fill=1, stroke=0)
        c.setFillColor(HexColor("#111827"))
        draw_text_center(c, label, x, 30.0, 5.8)

    c.saveState()
    c.translate(11.0, bottom + height / 2)
    c.rotate(90)
    c.setFillColor(HexColor("#374151"))
    c.setFont("Helvetica", 6.2)
    c.drawCentredString(0, 0, "Mean TM-score per cluster pair")
    c.restoreState()

    test_lookup = {(row["category_a"], row["category_b"]): row for row in tests}
    brackets = [
        ("within_cluster", "between_clusters_same_group", 0.91),
        ("within_cluster", "between_groups", 0.975),
        ("between_clusters_same_group", "between_groups", 0.82),
    ]
    x_lookup = {name: x for x, (name, _, _) in zip(x_positions, CATEGORIES, strict=True)}
    c.setStrokeColor(HexColor("#111827"))
    c.setFillColor(HexColor("#111827"))
    c.setLineWidth(0.65)
    for a, b, level in brackets:
        row = test_lookup[(a, b)]
        x1, x2 = x_lookup[a], x_lookup[b]
        y = y_coord(level, bottom, height)
        c.line(x1, y - 4, x1, y)
        c.line(x2, y - 4, x2, y)
        c.line(x1, y, x2, y)
        c.setFont("Helvetica", 4.8)
        c.drawCentredString((x1 + x2) / 2, y + 2.3, f"{row['significance']} Holm p={float(row['holm_adjusted_p']):.5f}")

    c.restoreState()
    if standalone:
        c.showPage()
        c.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    rows = read_csv(DATA_DIR / "cluster_pair_summary.csv")
    tests = read_csv(DATA_DIR / "category_significance.csv")
    panel_pdf = PLOT_DIR / "panel_c_similarity_reproduced.pdf"
    render_panel(rows, tests, panel_pdf)
    print(panel_pdf)


if __name__ == "__main__":
    main()
