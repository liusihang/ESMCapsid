#!/usr/bin/env python3
"""Render Figure 3B from the packaged exported taxonomy hierarchy tables."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "panel_b_taxonomy"
DEFAULT_OUTPUT = ROOT / "plots" / "panel_b_taxonomy_reproduced.svg"
DEFAULT_CLASSIFICATION = ROOT / "data" / "cluster_fold_classification.csv"

GROUP_COLORS = {
    "HK97-like": "#1D4ED8",
    "picorna-like": "#D97706",
    "NCLDV-like": "#16A34A",
    "BTV-like": "#0F766E",
    "ino-like": "#BE185D",
    "levi-like": "#7C3AED",
    "micro-like": "#64748B",
    "Circoviridae-like": "#D53F8C",
    "Geminiviridae-like": "#4A5568",
}
REALM_COLORS = {
    "Duplodnaviria": "#8AA4C7",
    "Monodnaviria": "#E4AA92",
    "Riboviria": "#C7A6D8",
    "Varidnaviria": "#8DC6B5",
    "Other_Realm": "#C8C8C8",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def point(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render(data_dir: Path, classification_csv: Path, output: Path) -> dict[str, object]:
    node_rows = read_csv(data_dir / "taxonomy_tree_nodes.csv")
    edge_rows = read_csv(data_dir / "taxonomy_tree_edges.csv")
    family_rows = read_csv(data_dir / "family_top1_cluster.csv")
    classification_rows = read_csv(classification_csv)

    # The packaged family table retains one audited stale label; final99 and the
    # submitted panel both classify C62 as HK97-like. Keep the source CSV intact.
    applied_corrections: list[str] = []
    for row in family_rows:
        if (
            row["family"] == "Mesyanzhinovviridae"
            and row["top1_cluster"] == "62"
            and row["manuscript_group"] == "ino-like"
        ):
            row["manuscript_group"] = "HK97-like"
            applied_corrections.append("Mesyanzhinovviridae [C62]: ino-like -> HK97-like")

    nodes = {row["node_id"]: row for row in node_rows}
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    parent: dict[str, str] = {}
    for row in edge_rows:
        parent_id = row["parent_id"]
        child_id = row["child_id"]
        if parent_id not in nodes or child_id not in nodes:
            raise ValueError(f"Edge references an unknown node: {parent_id} -> {child_id}")
        children[parent_id].append(child_id)
        parent[child_id] = parent_id

    roots = [node_id for node_id in nodes if node_id not in parent]
    if roots != ["root"]:
        raise ValueError(f"Expected one root named 'root', found {roots}")

    family_by_name = {row["family"]: row for row in family_rows}
    leaves: list[str] = []

    def visit(node_id: str) -> None:
        if not children[node_id]:
            leaves.append(node_id)
            return
        for child_id in children[node_id]:
            visit(child_id)

    visit("root")
    if not leaves:
        raise ValueError("Taxonomy hierarchy has no leaves")

    start_angle = math.radians(-165)
    end_angle = math.radians(165)
    leaf_angle = {
        node_id: start_angle + (end_angle - start_angle) * index / max(len(leaves) - 1, 1)
        for index, node_id in enumerate(leaves)
    }
    node_angle: dict[str, float] = {}

    def assign_angle(node_id: str) -> float:
        if node_id in leaf_angle:
            node_angle[node_id] = leaf_angle[node_id]
        else:
            child_angles = [assign_angle(child_id) for child_id in children[node_id]]
            node_angle[node_id] = sum(child_angles) / len(child_angles)
        return node_angle[node_id]

    assign_angle("root")

    width, height = 1900, 1500
    cx, cy = 760.0, 750.0
    max_radius = 590.0
    max_depth = max(int(row["depth"]) for row in node_rows)
    radius_step = max_radius / max(max_depth, 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.branch{fill:none;stroke-linecap:round}</style>",
        '<text x="35" y="48" font-size="34" font-weight="700" fill="#111827">B</text>',
        '<text x="760" y="42" text-anchor="middle" font-size="24" font-weight="700" fill="#111827">Taxonomic distribution of mapped capsid groups</text>',
    ]

    for depth in range(1, max_depth + 1):
        radius = radius_step * depth
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="none" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )

    for edge in edge_rows:
        parent_id = edge["parent_id"]
        child_id = edge["child_id"]
        parent_depth = int(nodes[parent_id]["depth"])
        child_depth = int(nodes[child_id]["depth"])
        parent_xy = point(cx, cy, radius_step * parent_depth, node_angle[parent_id])
        child_xy = point(cx, cy, radius_step * child_depth, node_angle[child_id])
        realm = nodes[child_id].get("top_level") or nodes[parent_id].get("top_level")
        family = family_by_name.get(nodes[child_id]["label"])
        color = (
            GROUP_COLORS.get(family["manuscript_group"], "#9CA3AF")
            if family
            else REALM_COLORS.get(realm, "#94A3B8")
        )
        tooltip = (
            f"{nodes[child_id]['path'] or nodes[child_id]['label']} | "
            f"{nodes[child_id]['rank']} | n={nodes[child_id]['n_sequences']}"
        )
        parts.append(
            f'<line class="branch" x1="{parent_xy[0]:.3f}" y1="{parent_xy[1]:.3f}" '
            f'x2="{child_xy[0]:.3f}" y2="{child_xy[1]:.3f}" '
            f'stroke="{color}" stroke-width="1.35" opacity="0.84"><title>{esc(tooltip)}</title></line>'
        )

    for node_id in leaves:
        node = nodes[node_id]
        family = family_by_name.get(node["label"])
        if not family:
            continue
        angle = node_angle[node_id]
        marker_xy = point(cx, cy, max_radius + 12, angle)
        color = GROUP_COLORS.get(family["manuscript_group"], "#9CA3AF")
        tooltip = (
            f"{family['family']} | C{family['top1_cluster']} | "
            f"{family['manuscript_group']} | n={family['n_sequences_in_tree']}"
        )
        parts.append(
            f'<circle cx="{marker_xy[0]:.3f}" cy="{marker_xy[1]:.3f}" r="4.2" '
            f'fill="{color}" stroke="#FFFFFF" stroke-width="1"><title>{esc(tooltip)}</title></circle>'
        )

    for realm_id in children["root"]:
        realm = nodes[realm_id]
        label_xy = point(cx, cy, radius_step + 42, node_angle[realm_id])
        parts.append(
            f'<text x="{label_xy[0]:.3f}" y="{label_xy[1]:.3f}" text-anchor="middle" '
            f'font-size="15" font-weight="700" fill="{REALM_COLORS.get(realm["label"], "#475569")}">'
            f'{esc(realm["label"])} ({esc(realm["n_sequences"])})</text>'
        )

    root = nodes["root"]
    parts.extend(
        [
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="52" fill="#F8FAFC" stroke="#64748B" stroke-width="1.5"/>',
            f'<text x="{cx:.2f}" y="{cy - 7:.2f}" text-anchor="middle" font-size="16" font-weight="700" fill="#111827">Mapped taxonomy</text>',
            f'<text x="{cx:.2f}" y="{cy + 17:.2f}" text-anchor="middle" font-size="14" fill="#475569">n={esc(root["n_sequences"])}</text>',
        ]
    )

    legend_x, legend_y = 1450, 180
    parts.append(
        f'<text x="{legend_x}" y="{legend_y - 34}" font-size="20" font-weight="700" fill="#111827">Capsid fold group</text>'
    )
    groups = {
        row["manuscript_group"]
        for row in family_rows
    } | {
        row["combined_fold"]
        for row in classification_rows
        if row["combined_fold"] not in {"", "Unknown"}
    }
    groups = sorted(groups, key=str.casefold)
    for index, group in enumerate(groups):
        y = legend_y + index * 34
        parts.append(f'<rect x="{legend_x}" y="{y - 13}" width="18" height="18" rx="2" fill="{GROUP_COLORS.get(group, "#9CA3AF")}"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{y + 1}" font-size="16" fill="#334155">{esc(group)}</text>')

    parts.append("</svg>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "leaves": len(leaves),
        "families": len(family_rows),
        "groups": groups,
        "root_sequences": int(root["n_sequences"]),
        "applied_corrections": applied_corrections,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = render(
        args.data_dir.resolve(),
        args.classification_csv.resolve(),
        args.output.resolve(),
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
