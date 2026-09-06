#!/usr/bin/env python3
"""Build the four Figure 4B subgroup dendrograms from the current fold map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


FIGURE4_ROOT = Path(__file__).resolve().parents[1]
GROUP_ORDER = ["HK97-like", "picorna-like", "NCLDV-like", "micro-like"]
GROUP_STEMS = {
    "HK97-like": "hk97",
    "picorna-like": "picorna",
    "NCLDV-like": "ncldv",
    "micro-like": "micro",
}
TREE_STEMS = {
    "HK97-like": "HK97_like",
    "picorna-like": "picorna_like",
    "NCLDV-like": "NCLDV_like",
    "micro-like": "micro_like",
}
SUBGROUP_COLORS = {
    1: "#E377C2",
    2: "#2CA02C",
    3: "#3B82F6",
    4: "#FF7F0E",
    5: "#D62728",
    6: "#8C564B",
}
MIXED_BRANCH_COLOR = "#4C72B0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Complete cluster cosine-similarity matrix in wide format.",
    )
    parser.add_argument(
        "--fold-map",
        type=Path,
        default=None,
        help="Current cluster-to-fold map used for the subgroup analysis.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=FIGURE4_ROOT / "data" / "panel_b_subgroups",
    )
    parser.add_argument(
        "--tree-dir",
        type=Path,
        default=(
            FIGURE4_ROOT / "images" / "figures_tree_square"
            if (FIGURE4_ROOT / "images").exists()
            else FIGURE4_ROOT / "plots" / "panel_b_subgroups"
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional second directory receiving the four public SVG/PNG assets.",
    )
    return parser.parse_args()


def default_input_paths() -> tuple[Path, Path]:
    raise FileNotFoundError(
        "No implicit complete inputs are used. Pass --matrix and --fold-map "
        "for full regeneration, or run without them to use the public group tables."
    )


def load_inputs(matrix_path: Path, fold_map_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    similarity = pd.read_csv(matrix_path)
    if "cluster" not in similarity.columns:
        raise ValueError(f"Missing 'cluster' column in {matrix_path}")
    similarity = similarity.set_index("cluster")
    similarity.index = similarity.index.astype(int)
    similarity.columns = similarity.columns.astype(int)
    if similarity.index.duplicated().any() or similarity.columns.duplicated().any():
        raise ValueError("The cosine matrix contains duplicate cluster identifiers")
    if set(similarity.index) != set(similarity.columns):
        raise ValueError("The cosine matrix row and column cluster sets differ")
    if not np.allclose(similarity.to_numpy(dtype=float), similarity.to_numpy(dtype=float).T, atol=1e-8):
        raise ValueError("The cosine matrix is not symmetric")

    fold_map = pd.read_csv(fold_map_path)
    required = {"cluster", "fold"}
    if not required.issubset(fold_map.columns):
        raise ValueError(f"Fold map must contain {sorted(required)}: {fold_map_path}")
    fold_map = fold_map[["cluster", "fold"]].copy()
    fold_map["cluster"] = fold_map["cluster"].astype(int)
    if fold_map["cluster"].duplicated().any():
        raise ValueError("The fold map contains duplicate clusters")
    if not set(fold_map["cluster"]).issubset(set(similarity.index)):
        raise ValueError("The fold map contains clusters absent from the cosine matrix")
    missing_groups = set(GROUP_ORDER) - set(fold_map["fold"])
    if missing_groups:
        raise ValueError(f"Fold map is missing Figure 4B groups: {sorted(missing_groups)}")
    return similarity, fold_map


def distance_submatrix(similarity: pd.DataFrame, clusters: list[int]) -> np.ndarray:
    values = similarity.loc[clusters, clusters].to_numpy(dtype=float)
    distance = np.clip(1.0 - values, 0.0, None)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def leaf_order(z: np.ndarray, clusters: list[int]) -> list[int]:
    order = dendrogram(z, labels=clusters, orientation="right", no_plot=True)["leaves"]
    return [clusters[index] for index in order]


def infer_subgroups(
    similarity: pd.DataFrame,
    fold_map: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, list[int]]:
    clusters = sorted(fold_map.loc[fold_map["fold"].eq(group), "cluster"].tolist())
    if len(clusters) == 1:
        assignment_table = pd.DataFrame(
            [
                {
                    "fold": group,
                    "cluster": clusters[0],
                    "auto_subgroup": 1,
                    "best_k": 1,
                    "silhouette": np.nan,
                    "dendrogram_leaf_rank": 0,
                }
            ]
        )
        scans = pd.DataFrame([{"fold": group, "k": 1, "silhouette": np.nan}])
        return assignment_table, scans, None, clusters
    if len(clusters) < 3:
        raise ValueError(f"Figure 4B group {group} has too few clusters: {clusters}")
    distance = distance_submatrix(similarity, clusters)
    z = linkage(squareform(distance, checks=False), method="average")
    order = leaf_order(z, clusters)

    scans = []
    best = None
    max_k = min(6, len(clusters) - 1)
    for k in range(2, max_k + 1):
        labels = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distance)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        scans.append({"fold": group, "k": k, "silhouette": score})
        if best is None or score > best[0] + 1e-12:
            best = (score, labels)
    if best is None:
        raise ValueError(f"Could not select a subgroup split for {group}")

    best_score, raw_labels = best
    first_position = {cluster: position for position, cluster in enumerate(order)}
    raw_order = sorted(
        set(int(label) for label in raw_labels),
        key=lambda label: min(first_position[clusters[i]] for i, value in enumerate(raw_labels) if value == label),
    )
    relabel = {raw_label: index + 1 for index, raw_label in enumerate(raw_order)}
    assignments = {cluster: relabel[int(raw_labels[index])] for index, cluster in enumerate(clusters)}
    rank = {cluster: index for index, cluster in enumerate(order)}

    assignment_table = pd.DataFrame(
        [
            {
                "fold": group,
                "cluster": cluster,
                "auto_subgroup": assignments[cluster],
                "best_k": len(raw_order),
                "silhouette": best_score,
                "dendrogram_leaf_rank": rank[cluster],
            }
            for cluster in clusters
        ]
    ).sort_values("cluster")
    return assignment_table, pd.DataFrame(scans), z, order


def load_public_group_inputs(
    data_dir: Path,
) -> list[tuple[str, pd.DataFrame, np.ndarray | None, list[int]]]:
    """Load only the non-sensitive per-group inputs shipped in the public package."""

    loaded = []
    for group in GROUP_ORDER:
        stem = GROUP_STEMS[group]
        matrix_path = data_dir / f"{stem}_cosine_similarity_matrix.csv"
        assignment_path = data_dir / f"{stem}_subgroup_assignment.csv"
        similarity = pd.read_csv(matrix_path)
        if "cluster" not in similarity.columns:
            raise ValueError(f"Missing 'cluster' column in {matrix_path}")
        similarity = similarity.set_index("cluster")
        similarity.index = similarity.index.astype(int)
        similarity.columns = similarity.columns.astype(int)
        clusters = sorted(similarity.index.tolist())
        if set(similarity.index) != set(similarity.columns):
            raise ValueError(f"The public matrix is not square: {matrix_path}")
        if not np.allclose(
            similarity.to_numpy(dtype=float),
            similarity.to_numpy(dtype=float).T,
            atol=1e-8,
        ):
            raise ValueError(f"The public matrix is not symmetric: {matrix_path}")
        assignment = pd.read_csv(assignment_path)
        required = {"fold", "cluster", "auto_subgroup"}
        if not required.issubset(assignment.columns):
            raise ValueError(
                f"Public assignment table must contain {sorted(required)}: {assignment_path}"
            )
        assignment["cluster"] = assignment["cluster"].astype(int)
        assignment["auto_subgroup"] = assignment["auto_subgroup"].astype(int)
        if set(assignment["cluster"]) != set(clusters):
            raise ValueError(f"Matrix and assignment cluster sets differ for {group}")
        if set(assignment["fold"]) != {group}:
            raise ValueError(f"Assignment table contains unexpected fold labels for {group}")
        distance = distance_submatrix(similarity, clusters)
        z = (
            None
            if len(clusters) == 1
            else linkage(squareform(distance, checks=False), method="average")
        )
        loaded.append((group, assignment.sort_values("cluster"), z, clusters))
    return loaded


def render_public_outputs(
    data_dir: Path,
    tree_dir: Path,
    plot_dir: Path | None,
) -> None:
    """Regenerate the four panel assets from the public per-group tables."""

    data_dir.mkdir(parents=True, exist_ok=True)
    tree_dir.mkdir(parents=True, exist_ok=True)
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
    for group, assignment, z, clusters in load_public_group_inputs(data_dir):
        stem = GROUP_STEMS[group]
        tree_stem = TREE_STEMS[group]
        output_paths = [
            data_dir / f"{stem}_subgroups.svg",
            data_dir / f"{stem}_subgroups.png",
            tree_dir / f"{tree_stem}_dendrogram_square.svg",
            tree_dir / f"{tree_stem}_dendrogram_square.png",
        ]
        if plot_dir is not None:
            output_paths.extend(
                [
                    plot_dir / f"{stem}_subgroups.svg",
                    plot_dir / f"{stem}_subgroups.png",
                ]
            )
        render_tree(group, assignment, z, clusters, output_paths)


def draw_subgroup_colored_dendrogram(
    ax: plt.Axes,
    z: np.ndarray,
    clusters: list[int],
    assignment: dict[int, int],
) -> None:
    """Draw child arms by child subgroup and shared joins by parent composition."""

    n_leaves = len(clusters)
    order = dendrogram(z, orientation="right", no_plot=True)["leaves"]
    node_x = {leaf: 0.0 for leaf in range(n_leaves)}
    node_y = {leaf: 5.0 + 10.0 * rank for rank, leaf in enumerate(order)}
    descendants = {leaf: {clusters[leaf]} for leaf in range(n_leaves)}

    def color_for(cluster_ids: set[int]) -> str:
        subgroups = {int(assignment[cluster]) for cluster in cluster_ids}
        return (
            SUBGROUP_COLORS[next(iter(subgroups))]
            if len(subgroups) == 1
            else MIXED_BRANCH_COLOR
        )

    for index, row in enumerate(z):
        left = int(row[0])
        right = int(row[1])
        distance = float(row[2])
        parent = n_leaves + index
        left_y = node_y[left]
        right_y = node_y[right]
        parent_descendants = descendants[left] | descendants[right]

        ax.plot(
            [node_x[left], distance],
            [left_y, left_y],
            color=color_for(descendants[left]),
            linewidth=1.5,
        )
        ax.plot(
            [node_x[right], distance],
            [right_y, right_y],
            color=color_for(descendants[right]),
            linewidth=1.5,
        )
        ax.plot(
            [distance, distance],
            [min(left_y, right_y), max(left_y, right_y)],
            color=color_for(parent_descendants),
            linewidth=1.5,
        )

        node_x[parent] = distance
        node_y[parent] = (left_y + right_y) / 2.0
        descendants[parent] = parent_descendants

    max_distance = float(np.max(z[:, 2]))
    ax.set_xlim(0.0, max_distance * 1.05 if max_distance > 0 else 1.0)
    ax.set_ylim(0.0, 10.0 * n_leaves)
    ax.set_yticks([node_y[leaf] for leaf in order])
    ax.set_yticklabels([str(clusters[leaf]) for leaf in order])


def render_tree(
    group: str,
    assignment_table: pd.DataFrame,
    z: np.ndarray | None,
    clusters: list[int],
    output_paths: list[Path],
) -> None:
    assignment = dict(zip(assignment_table["cluster"], assignment_table["auto_subgroup"]))

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    if z is None:
        cluster = clusters[0]
        ax.plot(
            [0.0, 0.05],
            [1.0, 1.0],
            color=SUBGROUP_COLORS[int(assignment[cluster])],
            linewidth=3.0,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.8, 1.2)
        ax.set_yticks([1.0])
        ax.set_yticklabels([str(cluster)])
    else:
        draw_subgroup_colored_dendrogram(ax, z, clusters, assignment)
    for tick in ax.get_ymajorticklabels():
        cluster = int(tick.get_text())
        tick.set_color(SUBGROUP_COLORS[int(assignment[cluster])])
        tick.set_fontsize(5.2)
    ax.set_title(group, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Cosine distance", fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, length=3)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    subgroup_count = int(assignment_table["auto_subgroup"].nunique())
    handles = [
        Line2D(
            [0],
            [0],
            color=SUBGROUP_COLORS[index],
            linewidth=3.0,
            label=f"subgroup {index}",
        )
        for index in range(1, subgroup_count + 1)
    ]
    ax.legend(
        handles=handles,
        title="Subgroup",
        frameon=False,
        loc="lower right",
        fontsize=7,
        title_fontsize=8,
        handlelength=1.5,
        borderpad=0.2,
        labelspacing=0.35,
    )
    fig.tight_layout(pad=0.6)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white")
    plt.close(fig)


def write_outputs(
    similarity: pd.DataFrame,
    fold_map: pd.DataFrame,
    data_dir: Path,
    tree_dir: Path,
    plot_dir: Path | None,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tree_dir.mkdir(parents=True, exist_ok=True)
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    all_assignments = []
    all_scans = []
    for group in GROUP_ORDER:
        assignment, scans, z, order = infer_subgroups(similarity, fold_map, group)
        stem = GROUP_STEMS[group]
        tree_stem = TREE_STEMS[group]
        clusters = sorted(assignment["cluster"].tolist())
        subgroup_matrix = similarity.loc[clusters, clusters].copy()
        subgroup_matrix.insert(0, "cluster", subgroup_matrix.index)
        subgroup_matrix.to_csv(data_dir / f"{stem}_cosine_similarity_matrix.csv", index=False)
        assignment.to_csv(data_dir / f"{stem}_subgroup_assignment.csv", index=False)
        scans.to_csv(data_dir / f"{stem}_k_selection_silhouette.csv", index=False)
        all_assignments.append(assignment)
        all_scans.append(scans)

        output_paths = [
            data_dir / f"{stem}_subgroups.svg",
            data_dir / f"{stem}_subgroups.png",
            tree_dir / f"{tree_stem}_dendrogram_square.svg",
            tree_dir / f"{tree_stem}_dendrogram_square.png",
        ]
        if plot_dir is not None:
            output_paths.extend(
                [
                    plot_dir / f"{stem}_subgroups.svg",
                    plot_dir / f"{stem}_subgroups.png",
                ]
            )
        render_tree(group, assignment, z, clusters, output_paths)

    combined = pd.concat(all_assignments, ignore_index=True)
    combined["fold"] = pd.Categorical(combined["fold"], categories=GROUP_ORDER, ordered=True)
    combined.sort_values(["fold", "cluster"]).to_csv(
        data_dir / "panel_b_subgroup_classification.csv", index=False
    )
    pd.concat(all_scans, ignore_index=True).to_csv(
        data_dir / "panel_b_k_selection_silhouette.csv", index=False
    )
    (data_dir / "README.md").write_text(
        "\n".join(
            [
                "# Figure 4B subgroup data",
                "",
                "The four dendrograms use the current kept-cluster cosine matrix and fold map.",
                "",
                "- Full regeneration accepts an upstream complete cluster cosine matrix and the Figure 4 fold map via --matrix and --fold-map.",
                "- Public regeneration uses the included per-group cosine matrices and assignment tables.",
                "- Fold-map source: Figure4/data/panel_b_subgroups/panel_b_fold_map.csv, synchronized from Figure3/data/cluster_fold_classification.csv combined_fold values.",
                "- Distance: 1 - cosine similarity",
                "- Linkage: average hierarchical linkage",
                "- Subgroup selection: best silhouette over k=2..6, limited by group size; singleton groups are retained as k=1 without a silhouette score",
                "",
                "The master classification table is panel_b_subgroup_classification.csv.",
                "The public package can regenerate the four dendrograms from the included per-group matrices and assignment tables without the complete upstream matrix.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if (args.matrix is None) != (args.fold_map is None):
        raise SystemExit("Pass both --matrix and --fold-map, or neither.")
    if args.matrix is not None and args.fold_map is not None:
        similarity, fold_map = load_inputs(args.matrix, args.fold_map)
        write_outputs(similarity, fold_map, args.data_dir, args.tree_dir, args.plot_dir)
        print("Figure 4B subgroup classification and four dendrograms written")
        return
    try:
        matrix_path, fold_map_path = default_input_paths()
    except FileNotFoundError:
        render_public_outputs(args.data_dir, args.tree_dir, args.plot_dir)
        print("Figure 4B subgroup dendrograms regenerated from public group tables")
        return
    similarity, fold_map = load_inputs(matrix_path, fold_map_path)
    write_outputs(similarity, fold_map, args.data_dir, args.tree_dir, args.plot_dir)
    print("Figure 4B subgroup classification and four dendrograms written")


if __name__ == "__main__":
    main()
