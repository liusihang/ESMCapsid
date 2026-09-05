from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import gemmi
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import combine_pvalues

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("default")
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"


ROOT = Path("/path/to/science_workspace")
ASSEMBLY_DIR = ROOT / "pdb_capsid_assemblies"
INPUT_DIR = ASSEMBLY_DIR / "group_core_motif_mapping_20260307"

MANIFEST = ASSEMBLY_DIR / "manifest.tsv"
ARCH_METRICS = INPUT_DIR / "updated_core_architecture_metrics_per_structure.csv"
UPDATED_REGIONS_ALL = INPUT_DIR / "updated_core_regions_merged_ranges_all_groups.csv"

PER_STRUCTURE_OUT = INPUT_DIR / "cluster_core_robust_evidence_per_structure.csv"
GROUP_SUMMARY_OUT = INPUT_DIR / "cluster_core_robust_evidence_group_summary.csv"
FIG_OUT = INPUT_DIR / "pngs" / "cluster_core_robust_evidence.png"

GROUP_ORDER = ["HK97_like", "NCLDV_like", "Pico_like"]
GROUP_COLORS = {
    "HK97_like": "#1f4e79",
    "NCLDV_like": "#8c2d04",
    "Pico_like": "#2b6c4a",
}

COORD_FILE_OVERRIDES = {
    "5J7V": ASSEMBLY_DIR / "raw" / "5J7V-entry.cif.gz",
}

NEIGHBORHOOD_RADIUS = 15.0
CONTACT_CUTOFF = 8.0
BACKBONE_EXCLUSION = 2
NULL_ITERATIONS = 300
RANDOM_SEED = 7
MATCH_TOLERANCES = [0.03, 0.05, 0.08, 0.12, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cluster-core structural evidence against matched nulls."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--coord-override-5j7v", type=Path)
    parser.add_argument("--null-iterations", type=int, default=NULL_ITERATIONS)
    return parser.parse_args()


@dataclass
class LocalGraphData:
    coord_file: Path
    rep_chain: str
    available_positions: list[int]
    position_to_node_index: dict[int, int]
    position_to_rep_array_index: dict[int, int]
    position_to_radial: dict[int, float]
    degree_by_position: dict[int, float]
    clustering_by_position: dict[int, float]
    coreness_by_position: dict[int, float]
    pairwise_distance_matrix: np.ndarray
    n_local_nodes: int
    n_local_edges: int


def seq_id_key(seq_id: str) -> tuple[str, int]:
    match = re.match(r"^([0-9A-Za-z]{4})_(\d+)_Chain[s]?$", seq_id)
    if not match:
        raise ValueError(f"Unsupported Seq_ID: {seq_id}")
    return match.group(1), int(match.group(2))


def contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    ordered = sorted(set(int(pos) for pos in positions))
    runs: list[tuple[int, int]] = []
    start = ordered[0]
    prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev))
        start = pos
        prev = pos
    runs.append((start, prev))
    return runs


def fisher_pvalue(values: pd.Series) -> float:
    clean = values.dropna().clip(lower=1e-300)
    if clean.empty:
        return float("nan")
    return float(combine_pvalues(clean, method="fisher").pvalue)


def empirical_summary(
    observed: float, null_values: list[float], higher_is_more_extreme: bool
) -> tuple[float, float, float]:
    null_array = np.asarray(null_values, dtype=float)
    null_mean = float(np.mean(null_array))
    null_std = float(np.std(null_array, ddof=1)) if len(null_array) > 1 else 0.0
    z_score = (observed - null_mean) / max(null_std, 1e-9)
    if higher_is_more_extreme:
        empirical_p = (float(np.sum(null_array >= observed)) + 1.0) / (
            len(null_array) + 1.0
        )
    else:
        empirical_p = (float(np.sum(null_array <= observed)) + 1.0) / (
            len(null_array) + 1.0
        )
    return null_mean, float(z_score), float(empirical_p)


def metric_values(data: LocalGraphData, positions: list[int]) -> dict[str, float]:
    node_indices = [data.position_to_node_index[pos] for pos in positions]
    rep_array_indices = [data.position_to_rep_array_index[pos] for pos in positions]

    degree_values = np.asarray(
        [data.degree_by_position[pos] for pos in positions], dtype=float
    )
    clustering_values = np.asarray(
        [data.clustering_by_position[pos] for pos in positions], dtype=float
    )
    coreness_values = np.asarray(
        [data.coreness_by_position[pos] for pos in positions], dtype=float
    )
    radial_values = np.asarray(
        [data.position_to_radial[pos] for pos in positions], dtype=float
    )

    pairwise_sub = data.pairwise_distance_matrix[
        np.ix_(rep_array_indices, rep_array_indices)
    ]
    upper = pairwise_sub[np.triu_indices(len(rep_array_indices), 1)]
    pairwise_median = float(np.median(upper)) if len(upper) else float("nan")

    return {
        "mean_degree": float(np.mean(degree_values)),
        "mean_clustering": float(np.mean(clustering_values)),
        "mean_coreness": float(np.mean(coreness_values)),
        "mean_radial": float(np.mean(radial_values)),
        "pairwise_median": pairwise_median,
    }


def load_local_graph(coord_file: Path, entity_id: int) -> LocalGraphData:
    structure = gemmi.read_structure(str(coord_file))
    model = structure[0]

    entity = None
    for item in structure.entities:
        if item.entity_type.name == "Polymer" and item.name == str(entity_id):
            entity = item
            break
    if entity is None:
        raise KeyError(f"Entity {entity_id} not found in {coord_file}")

    rep_subchain = entity.subchains[0]
    rep_chain = None
    rep_positions: list[tuple[int, np.ndarray]] = []
    all_ca_points: list[np.ndarray] = []

    for chain in model:
        current_rep: list[tuple[int, np.ndarray]] = []
        for residue in chain:
            if residue.het_flag != "A":
                continue
            atom = residue.find_atom("CA", "\0")
            if atom is None:
                continue
            point = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            all_ca_points.append(point)
            label_seq = getattr(residue, "label_seq", None)
            if residue.subchain == rep_subchain and label_seq is not None:
                current_rep.append((int(label_seq), point))
        if current_rep and rep_chain is None:
            rep_chain = chain.name
            rep_positions = current_rep

    if rep_chain is None or not rep_positions:
        raise RuntimeError(
            f"No representative chain found for {coord_file} entity {entity_id}"
        )

    all_ca = np.vstack(all_ca_points)
    assembly_center = np.mean(all_ca, axis=0)
    all_distances = np.linalg.norm(all_ca - assembly_center, axis=1)
    min_distance = float(np.min(all_distances))
    max_distance = float(np.max(all_distances))
    radial_denom = max(max_distance - min_distance, 1e-6)

    rep_labels = [label for label, _ in rep_positions]
    rep_coords = np.vstack([coords for _, coords in rep_positions])
    rep_tree = cKDTree(rep_coords)
    rep_radials = {
        label: float(
            (np.linalg.norm(coords - assembly_center) - min_distance) / radial_denom
        )
        for label, coords in rep_positions
    }

    local_nodes: list[tuple[str, int, np.ndarray]] = []
    for chain in model:
        coords: list[np.ndarray] = []
        labels: list[int] = []
        for residue in chain:
            if residue.het_flag != "A":
                continue
            atom = residue.find_atom("CA", "\0")
            label_seq = getattr(residue, "label_seq", None)
            if atom is None or label_seq is None:
                continue
            coords.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float))
            labels.append(int(label_seq))
        if not coords:
            continue
        coord_array = np.vstack(coords)
        if chain.name == rep_chain:
            min_dists = np.zeros(len(coord_array), dtype=float)
        else:
            min_dists = rep_tree.query(coord_array, k=1)[0]
        for label, coord, min_dist in zip(labels, coord_array, min_dists):
            if chain.name == rep_chain or min_dist <= NEIGHBORHOOD_RADIUS:
                local_nodes.append((chain.name, label, coord))

    local_coords = np.vstack([coord for _, _, coord in local_nodes])
    local_tree = cKDTree(local_coords)
    contact_pairs = local_tree.query_pairs(CONTACT_CUTOFF)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(local_nodes)))
    for idx_a, idx_b in contact_pairs:
        chain_a, label_a, _ = local_nodes[idx_a]
        chain_b, label_b, _ = local_nodes[idx_b]
        if chain_a == chain_b and abs(label_a - label_b) <= BACKBONE_EXCLUSION:
            continue
        graph.add_edge(idx_a, idx_b)

    degree_map = dict(graph.degree())
    clustering_map = nx.clustering(graph)
    coreness_map = nx.core_number(graph)

    rep_local_positions = sorted(
        label for chain, label, _ in local_nodes if chain == rep_chain
    )
    position_to_node_index = {
        label: idx
        for idx, (chain, label, _) in enumerate(local_nodes)
        if chain == rep_chain
    }
    position_to_rep_array_index = {label: idx for idx, label in enumerate(rep_labels)}

    degree_by_position = {
        pos: float(degree_map[position_to_node_index[pos]])
        for pos in rep_local_positions
    }
    clustering_by_position = {
        pos: float(clustering_map[position_to_node_index[pos]])
        for pos in rep_local_positions
    }
    coreness_by_position = {
        pos: float(coreness_map[position_to_node_index[pos]])
        for pos in rep_local_positions
    }

    pairwise_distance_matrix = np.linalg.norm(
        rep_coords[:, None, :] - rep_coords[None, :, :], axis=2
    )

    return LocalGraphData(
        coord_file=coord_file,
        rep_chain=rep_chain,
        available_positions=rep_local_positions,
        position_to_node_index=position_to_node_index,
        position_to_rep_array_index=position_to_rep_array_index,
        position_to_radial=rep_radials,
        degree_by_position=degree_by_position,
        clustering_by_position=clustering_by_position,
        coreness_by_position=coreness_by_position,
        pairwise_distance_matrix=pairwise_distance_matrix,
        n_local_nodes=int(len(local_nodes)),
        n_local_edges=int(graph.number_of_edges()),
    )


def updated_region_positions(
    region_df: pd.DataFrame, group: str, seq_id: str
) -> list[int]:
    subset = region_df[(region_df["group"] == group) & (region_df["seq_id"] == seq_id)]
    positions: set[int] = set()
    for start, end in zip(subset["start"], subset["end"]):
        positions.update(range(int(start), int(end) + 1))
    return sorted(positions)


def sample_matched_positions(
    available_positions: list[int],
    core_positions: list[int],
    radial_lookup: dict[int, float],
    rng: np.random.Generator,
) -> tuple[list[list[int]], list[float]]:
    available_set = set(available_positions)
    core_set = set(core_positions)
    segments = contiguous_runs(core_positions)

    segment_specs: list[dict[str, float | int]] = []
    for start, end in segments:
        positions = list(range(start, end + 1))
        segment_specs.append(
            {
                "length": int(end - start + 1),
                "target_radial": float(
                    np.median([radial_lookup[pos] for pos in positions])
                ),
            }
        )

    segment_candidates: list[list[tuple[int, tuple[int, ...], float]]] = []
    for spec in segment_specs:
        length = int(spec["length"])
        candidates: list[tuple[int, tuple[int, ...], float]] = []
        for start in available_positions:
            segment = tuple(range(start, start + length))
            if not all(pos in available_set and pos not in core_set for pos in segment):
                continue
            median_radial = float(np.median([radial_lookup[pos] for pos in segment]))
            candidates.append((start, segment, median_radial))
        if not candidates:
            raise RuntimeError(f"No candidate segments for length {length}")
        segment_candidates.append(candidates)

    sampled_position_sets: list[list[int]] = []
    radial_match_errors: list[float] = []
    max_attempts = NULL_ITERATIONS * 8
    attempts = 0

    while len(sampled_position_sets) < NULL_ITERATIONS and attempts < max_attempts:
        attempts += 1
        order = sorted(
            range(len(segment_specs)),
            key=lambda idx: (-int(segment_specs[idx]["length"]), float(rng.random())),
        )
        used_positions: set[int] = set()
        sampled_segments: list[tuple[int, ...]] = []
        sampled_errors: list[float] = []
        failed = False

        for spec_idx in order:
            spec = segment_specs[spec_idx]
            target_radial = float(spec["target_radial"])
            chosen_segment: tuple[int, ...] | None = None
            chosen_error = float("nan")

            for tolerance in MATCH_TOLERANCES:
                pool = [
                    (segment, abs(candidate_radial - target_radial))
                    for _, segment, candidate_radial in segment_candidates[spec_idx]
                    if abs(candidate_radial - target_radial) <= tolerance
                    and all(pos not in used_positions for pos in segment)
                ]
                if pool:
                    pick_idx = int(rng.integers(len(pool)))
                    chosen_segment, chosen_error = pool[pick_idx]
                    break

            if chosen_segment is None:
                fallback_pool = [
                    (segment, abs(candidate_radial - target_radial))
                    for _, segment, candidate_radial in segment_candidates[spec_idx]
                    if all(pos not in used_positions for pos in segment)
                ]
                if not fallback_pool:
                    failed = True
                    break
                errors = np.asarray([item[1] for item in fallback_pool], dtype=float)
                min_error = float(np.min(errors))
                close_choices = [
                    item
                    for item in fallback_pool
                    if math.isclose(item[1], min_error) or item[1] == min_error
                ]
                pick_idx = int(rng.integers(len(close_choices)))
                chosen_segment, chosen_error = close_choices[pick_idx]

            sampled_segments.append(chosen_segment)
            sampled_errors.append(chosen_error)
            used_positions.update(chosen_segment)

        if failed:
            continue

        sampled_positions = sorted(
            pos for segment in sampled_segments for pos in segment
        )
        sampled_position_sets.append(sampled_positions)
        radial_match_errors.append(
            float(np.mean(sampled_errors)) if sampled_errors else float("nan")
        )

    if len(sampled_position_sets) < max(NULL_ITERATIONS // 2, 50):
        raise RuntimeError(
            f"Insufficient matched-null samples: {len(sampled_position_sets)} / {NULL_ITERATIONS}"
        )

    return sampled_position_sets, radial_match_errors


def plot_metric_panel(
    ax: plt.Axes,
    detail_df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    positive_is_stronger: bool,
) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    for idx, group in enumerate(GROUP_ORDER):
        group_df = detail_df[detail_df["group"] == group]
        if group_df.empty:
            continue
        y = np.full(len(group_df), idx, dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(group_df))
        color = GROUP_COLORS[group]
        ax.scatter(
            group_df[column],
            y + jitter,
            s=46,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter(
            [group_df[column].mean()],
            [idx],
            marker="D",
            s=92,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels(GROUP_ORDER)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, alpha=0.8)

    label = (
        "stronger than matched null"
        if positive_is_stronger
        else "more compact than matched null"
    )
    ax.annotate(
        label,
        xy=(0.99 if positive_is_stronger else 0.01, 0.02),
        xycoords="axes fraction",
        ha="right" if positive_is_stronger else "left",
        va="bottom",
        fontsize=9,
        color="#444444",
    )


def make_figure(detail_df: pd.DataFrame) -> None:
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    plot_metric_panel(
        axes[0, 0],
        detail_df,
        column="degree_z",
        title="Local Contact Degree",
        xlabel="z-score vs radial/segment-matched null",
        positive_is_stronger=True,
    )
    plot_metric_panel(
        axes[0, 1],
        detail_df,
        column="coreness_z",
        title="Shell Contact Coreness",
        xlabel="z-score vs radial/segment-matched null",
        positive_is_stronger=True,
    )
    plot_metric_panel(
        axes[1, 0],
        detail_df,
        column="clustering_z",
        title="Neighborhood Mesh Clustering",
        xlabel="z-score vs radial/segment-matched null",
        positive_is_stronger=True,
    )
    plot_metric_panel(
        axes[1, 1],
        detail_df,
        column="compactness_z",
        title="3D Compactness",
        xlabel="-z-score of pairwise distance vs matched null",
        positive_is_stronger=True,
    )

    fig.suptitle(
        "Cluster-level lineage core motifs: robust local-network evidence",
        fontsize=15,
        y=1.02,
    )
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    global INPUT_DIR, MANIFEST, ARCH_METRICS, UPDATED_REGIONS_ALL
    global PER_STRUCTURE_OUT, GROUP_SUMMARY_OUT, FIG_OUT
    global COORD_FILE_OVERRIDES, NULL_ITERATIONS

    args = parse_args()
    INPUT_DIR = args.input_dir.expanduser().resolve()
    MANIFEST = args.manifest.expanduser().resolve()
    ARCH_METRICS = INPUT_DIR / "updated_core_architecture_metrics_per_structure.csv"
    UPDATED_REGIONS_ALL = (
        INPUT_DIR / "updated_core_regions_merged_ranges_all_groups.csv"
    )
    PER_STRUCTURE_OUT = INPUT_DIR / "cluster_core_robust_evidence_per_structure.csv"
    GROUP_SUMMARY_OUT = INPUT_DIR / "cluster_core_robust_evidence_group_summary.csv"
    FIG_OUT = INPUT_DIR / "pngs" / "cluster_core_robust_evidence.png"
    COORD_FILE_OVERRIDES = {}
    if args.coord_override_5j7v:
        COORD_FILE_OVERRIDES["5J7V"] = args.coord_override_5j7v.expanduser().resolve()
    if args.null_iterations < 50:
        raise ValueError("--null-iterations must be at least 50")
    NULL_ITERATIONS = args.null_iterations

    rng = np.random.default_rng(RANDOM_SEED)

    manifest = pd.read_csv(MANIFEST, sep="\t")
    arch_df = (
        pd.read_csv(ARCH_METRICS)
        .sort_values(["group", "seq_id"])
        .reset_index(drop=True)
    )
    region_df = pd.read_csv(UPDATED_REGIONS_ALL)
    coord_lookup = dict(zip(manifest["pdb_id"], manifest["assembly_file"]))

    detail_rows: list[dict[str, float | int | str]] = []
    cache: dict[tuple[str, int], LocalGraphData] = {}

    for _, arch_row in arch_df.iterrows():
        group = str(arch_row["group"])
        seq_id = str(arch_row["seq_id"])
        pdb_id, entity_id = seq_id_key(seq_id)
        coord_file = COORD_FILE_OVERRIDES.get(pdb_id, Path(coord_lookup[pdb_id]))
        cache_key = (pdb_id, entity_id)
        if cache_key not in cache:
            cache[cache_key] = load_local_graph(coord_file, entity_id)
        local_data = cache[cache_key]

        raw_core_positions = updated_region_positions(region_df, group, seq_id)
        core_source = "updated_projected_motif_ids_exact_structure_intersection"

        core_positions = sorted(
            pos
            for pos in raw_core_positions
            if pos in local_data.position_to_node_index
        )
        if len(core_positions) < 3:
            raise RuntimeError(
                f"Too few mapped core positions for {group} {seq_id}: {len(core_positions)}"
            )

        null_samples, radial_errors = sample_matched_positions(
            available_positions=local_data.available_positions,
            core_positions=core_positions,
            radial_lookup=local_data.position_to_radial,
            rng=rng,
        )

        observed = metric_values(local_data, core_positions)
        null_metric_rows = [
            metric_values(local_data, sample) for sample in null_samples
        ]

        degree_null = [row["mean_degree"] for row in null_metric_rows]
        clustering_null = [row["mean_clustering"] for row in null_metric_rows]
        coreness_null = [row["mean_coreness"] for row in null_metric_rows]
        pairwise_null = [row["pairwise_median"] for row in null_metric_rows]

        degree_null_mean, degree_z, degree_p = empirical_summary(
            observed["mean_degree"], degree_null, True
        )
        clustering_null_mean, clustering_z, clustering_p = empirical_summary(
            observed["mean_clustering"], clustering_null, True
        )
        coreness_null_mean, coreness_z, coreness_p = empirical_summary(
            observed["mean_coreness"], coreness_null, True
        )
        pairwise_null_mean, pairwise_z, pairwise_p = empirical_summary(
            observed["pairwise_median"], pairwise_null, False
        )

        detail_rows.append(
            {
                "group": group,
                "seq_id": seq_id,
                "pdb_id": pdb_id,
                "entity_id": entity_id,
                "coord_file": str(coord_file),
                "rep_chain": local_data.rep_chain,
                "core_source": core_source,
                "n_core_res": int(len(core_positions)),
                "n_segments": int(len(contiguous_runs(core_positions))),
                "n_local_nodes": int(local_data.n_local_nodes),
                "n_local_edges": int(local_data.n_local_edges),
                "n_null_samples": int(len(null_samples)),
                "null_mean_segment_radial_absdiff": float(np.mean(radial_errors)),
                "core_mean_degree": observed["mean_degree"],
                "null_mean_degree": degree_null_mean,
                "degree_ratio": observed["mean_degree"] / max(degree_null_mean, 1e-9),
                "degree_z": degree_z,
                "degree_empirical_p_higher": degree_p,
                "core_mean_clustering": observed["mean_clustering"],
                "null_mean_clustering": clustering_null_mean,
                "clustering_ratio": observed["mean_clustering"]
                / max(clustering_null_mean, 1e-9),
                "clustering_z": clustering_z,
                "clustering_empirical_p_higher": clustering_p,
                "core_mean_coreness": observed["mean_coreness"],
                "null_mean_coreness": coreness_null_mean,
                "coreness_ratio": observed["mean_coreness"]
                / max(coreness_null_mean, 1e-9),
                "coreness_z": coreness_z,
                "coreness_empirical_p_higher": coreness_p,
                "core_pairwise_median": observed["pairwise_median"],
                "null_pairwise_median_mean": pairwise_null_mean,
                "pairwise_ratio": observed["pairwise_median"]
                / max(pairwise_null_mean, 1e-9),
                "pairwise_reduction_pct": (
                    1.0 - observed["pairwise_median"] / max(pairwise_null_mean, 1e-9)
                )
                * 100.0,
                "pairwise_z": pairwise_z,
                "pairwise_empirical_p_lower": pairwise_p,
                "compactness_z": -pairwise_z,
            }
        )

    detail_df = (
        pd.DataFrame(detail_rows)
        .sort_values(["group", "seq_id"])
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, float | int | str]] = []
    for group in GROUP_ORDER:
        group_df = detail_df[detail_df["group"] == group]
        if group_df.empty:
            continue
        summary_rows.append(
            {
                "group": group,
                "n_structures": int(len(group_df)),
                "mean_n_core_res": float(group_df["n_core_res"].mean()),
                "mean_n_segments": float(group_df["n_segments"].mean()),
                "mean_n_local_nodes": float(group_df["n_local_nodes"].mean()),
                "mean_null_segment_radial_absdiff": float(
                    group_df["null_mean_segment_radial_absdiff"].mean()
                ),
                "mean_degree_ratio": float(group_df["degree_ratio"].mean()),
                "mean_degree_z": float(group_df["degree_z"].mean()),
                "fisher_degree_p": fisher_pvalue(group_df["degree_empirical_p_higher"]),
                "n_degree_p_lt_0_05": int(
                    (group_df["degree_empirical_p_higher"] < 0.05).sum()
                ),
                "mean_clustering_ratio": float(group_df["clustering_ratio"].mean()),
                "mean_clustering_z": float(group_df["clustering_z"].mean()),
                "fisher_clustering_p": fisher_pvalue(
                    group_df["clustering_empirical_p_higher"]
                ),
                "n_clustering_p_lt_0_05": int(
                    (group_df["clustering_empirical_p_higher"] < 0.05).sum()
                ),
                "mean_coreness_ratio": float(group_df["coreness_ratio"].mean()),
                "mean_coreness_z": float(group_df["coreness_z"].mean()),
                "fisher_coreness_p": fisher_pvalue(
                    group_df["coreness_empirical_p_higher"]
                ),
                "n_coreness_p_lt_0_05": int(
                    (group_df["coreness_empirical_p_higher"] < 0.05).sum()
                ),
                "mean_pairwise_ratio": float(group_df["pairwise_ratio"].mean()),
                "mean_pairwise_reduction_pct": float(
                    group_df["pairwise_reduction_pct"].mean()
                ),
                "mean_pairwise_z": float(group_df["pairwise_z"].mean()),
                "fisher_pairwise_p": fisher_pvalue(
                    group_df["pairwise_empirical_p_lower"]
                ),
                "n_pairwise_p_lt_0_05": int(
                    (group_df["pairwise_empirical_p_lower"] < 0.05).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    PER_STRUCTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(PER_STRUCTURE_OUT, index=False)
    summary_df.to_csv(GROUP_SUMMARY_OUT, index=False)
    make_figure(detail_df)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False))
    print(f"\nWrote {PER_STRUCTURE_OUT}")
    print(f"Wrote {GROUP_SUMMARY_OUT}")
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
