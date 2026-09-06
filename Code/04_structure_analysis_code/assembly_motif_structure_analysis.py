from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.special import rel_entr

MANIFEST = Path("/path/to/science_workspace/pdb_capsid_assemblies/manifest.tsv")
SEQ_REPORT = Path(
    "/path/to/science_workspace/pdb_capsid_assemblies/sae_report_20260307/final_sequence_predictions.csv"
)
TOKEN_REPORT = Path(
    "/path/to/science_workspace/pdb_capsid_assemblies/sae_report_20260307/final_token_semantic_motif.csv"
)
OUTPUT_DIR = Path("/path/to/science_workspace/pdb_capsid_assemblies/analysis_20260307")

BINS = np.linspace(0.0, 1.0, 21)
LOW_JSD = 0.08
MEDIUM_JSD = 0.16

TARGET_FAMILIES: dict[str, list[tuple[str, int]]] = {
    "HK97_MCP_all": [("1OHG", 1), ("2XYY", 1), ("3QPR", 1), ("5UU5", 1), ("5ZAP", 1)],
    "HK97_MCP_phage": [("1OHG", 1), ("2XYY", 1), ("3QPR", 1), ("5UU5", 1)],
    "NCLDV_MCP": [("1M4X", 1), ("5J7V", 1), ("6NCL", 14), ("3J31", 3)],
    "Pico_VP1": [("1B35", 1), ("1HXS", 1), ("4GB3", 1), ("4RHV", 1), ("5WTE", 1)],
    "Pico_VP2": [("1B35", 2), ("1HXS", 2), ("4GB3", 2), ("4RHV", 2), ("5WTE", 2)],
    "Pico_VP3": [("1B35", 3), ("1HXS", 3), ("4GB3", 3), ("4RHV", 3), ("5WTE", 3)],
}

COORD_FILE_OVERRIDES = {
    "5J7V": Path(
        "/path/to/science_workspace/pdb_capsid_assemblies/raw/5J7V-entry.cif.gz"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze conserved semantic motifs on capsid assemblies."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sequence-report", required=True, type=Path)
    parser.add_argument("--token-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coord-override-5j7v", type=Path)
    return parser.parse_args()


@dataclass
class EntityCoords:
    pdb_id: str
    entity_id: int
    rep_chain: str
    coord_path: Path
    label_seq_to_residue: dict[int, gemmi.Residue]
    radial_stats: dict[int, tuple[float, float]]


def seq_id_key(seq_id: str) -> tuple[str, int]:
    match = re.match(r"^([0-9A-Za-z]{4})_(\d+)_Chain", seq_id)
    if not match:
        raise ValueError(f"Unrecognized Seq_ID format: {seq_id}")
    return match.group(1), int(match.group(2))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-300)
    q = q / (q.sum() + 1e-300)
    m = (p + q) / 2.0
    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))


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


def radial_class(value: float) -> str:
    if math.isnan(value):
        return "na"
    if value < 0.33:
        return "inner"
    if value < 0.67:
        return "middle"
    return "outer"


def load_entity_coords(coord_path: Path, pdb_id: str, entity_id: int) -> EntityCoords:
    structure = gemmi.read_structure(str(coord_path))
    model = structure[0]

    entity = None
    for item in structure.entities:
        if item.entity_type.name == "Polymer" and item.name == str(entity_id):
            entity = item
            break
    if entity is None:
        raise KeyError(f"Entity {entity_id} not found in {coord_path}")

    rep_subchain = entity.subchains[0]
    rep_chain_name = None
    label_seq_to_residue: dict[int, gemmi.Residue] = {}
    for chain in model:
        current = {}
        for residue in chain:
            if residue.het_flag != "A" or residue.subchain != rep_subchain:
                continue
            label_seq = getattr(residue, "label_seq", None)
            if label_seq is not None:
                current[int(label_seq)] = residue
        if current:
            rep_chain_name = chain.name
            label_seq_to_residue = current
            break
    if rep_chain_name is None:
        raise KeyError(f"Subchain {rep_subchain} not found in {coord_path}")

    centers = []
    residue_points: list[tuple[str, int, np.ndarray]] = []
    for chain in model:
        for residue in chain:
            if residue.het_flag != "A":
                continue
            atom = residue.find_atom("CA", "\0")
            if atom is None:
                continue
            pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            centers.append(pos)
            label_seq = getattr(residue, "label_seq", None)
            if label_seq is not None:
                residue_points.append((chain.name, int(label_seq), pos))
    if not centers:
        assembly_center = np.zeros(3, dtype=float)
    else:
        assembly_center = np.mean(np.vstack(centers), axis=0)

    distances = []
    keyed_points = []
    for chain_name, label_seq, pos in residue_points:
        dist = float(np.linalg.norm(pos - assembly_center))
        distances.append(dist)
        keyed_points.append((chain_name, label_seq, dist))
    ranked = sorted(distances)
    radial_stats: dict[int, tuple[float, float]] = {}
    if ranked:
        min_dist = ranked[0]
        max_dist = ranked[-1]
        denom = max(max_dist - min_dist, 1e-6)
        for chain_name, label_seq, dist in keyed_points:
            if chain_name != rep_chain_name:
                continue
            frac = (dist - min_dist) / denom
            percentile = float(
                np.searchsorted(ranked, dist, side="right") / len(ranked)
            )
            radial_stats[label_seq] = (frac, percentile)

    return EntityCoords(
        pdb_id=pdb_id,
        entity_id=entity_id,
        rep_chain=rep_chain_name,
        coord_path=coord_path,
        label_seq_to_residue=label_seq_to_residue,
        radial_stats=radial_stats,
    )


def main() -> None:
    global MANIFEST, SEQ_REPORT, TOKEN_REPORT, OUTPUT_DIR, COORD_FILE_OVERRIDES

    args = parse_args()
    MANIFEST = args.manifest.expanduser().resolve()
    SEQ_REPORT = args.sequence_report.expanduser().resolve()
    TOKEN_REPORT = args.token_report.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    COORD_FILE_OVERRIDES = {}
    if args.coord_override_5j7v:
        COORD_FILE_OVERRIDES["5J7V"] = args.coord_override_5j7v.expanduser().resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, sep="\t")
    seq_df = pd.read_csv(SEQ_REPORT)
    token_df = pd.read_csv(TOKEN_REPORT)
    seq_df["key"] = seq_df["Seq_ID"].map(seq_id_key)
    seq_lookup = dict(zip(seq_df["key"], seq_df["Seq_ID"]))
    token_groups = {
        seq_id: grp.sort_values("sequence_position").copy()
        for seq_id, grp in token_df.groupby("Seq_ID")
    }

    entity_cache: dict[tuple[str, int], EntityCoords] = {}
    summary_rows = []
    motif_rows = []
    site_rows = []

    for family_name, targets in TARGET_FAMILIES.items():
        seq_ids = []
        for target in targets:
            seq_id = seq_lookup.get(target)
            if seq_id is None:
                raise KeyError(f"Missing Seq_ID for {target}")
            seq_ids.append(seq_id)

        motif_sets = [
            set(token_groups[seq_id]["semantic_motif_label"]) for seq_id in seq_ids
        ]
        shared_motifs = sorted(set.intersection(*motif_sets)) if motif_sets else []

        family_motif_rows = []
        low_count = 0
        med_count = 0
        high_count = 0

        for motif in shared_motifs:
            histograms = []
            mean_positions = []
            all_relative_positions = []
            representative_runs = {}
            sequence_positions_by_seq = {}

            per_seq_mode_positions = []
            for seq_id in seq_ids:
                seq_tokens = token_groups[seq_id]
                motif_tokens = seq_tokens[seq_tokens["semantic_motif_label"] == motif]
                positions = motif_tokens["sequence_position"].astype(int).tolist()
                sequence_positions_by_seq[seq_id] = positions
                seq_len = int(seq_tokens["sequence_position"].max())
                rel_positions = (
                    motif_tokens["sequence_position"].to_numpy(dtype=float) - 1.0
                ) / seq_len
                hist, _ = np.histogram(rel_positions, bins=BINS)
                histograms.append(hist)
                mean_positions.append(float(rel_positions.mean()))
                all_relative_positions.extend(rel_positions.tolist())
                runs = contiguous_runs(positions)
                if runs:
                    run_centers = [
                        ((start + end) / 2.0 - 1.0) / seq_len for start, end in runs
                    ]
                    representative_runs[seq_id] = (runs, run_centers)
                    per_seq_mode_positions.append(float(np.median(run_centers)))

            pairwise_jsd = [jsd(a, b) for a, b in combinations(histograms, 2)]
            mean_jsd = float(np.mean(pairwise_jsd)) if pairwise_jsd else 0.0
            median_jsd = float(np.median(pairwise_jsd)) if pairwise_jsd else 0.0
            mean_position_span = (
                float(max(mean_positions) - min(mean_positions))
                if mean_positions
                else float("nan")
            )
            family_mean_relpos = (
                float(np.mean(all_relative_positions))
                if all_relative_positions
                else float("nan")
            )

            if mean_jsd <= LOW_JSD:
                divergence_class = "low"
                low_count += 1
            elif mean_jsd <= MEDIUM_JSD:
                divergence_class = "medium"
                med_count += 1
            else:
                divergence_class = "high"
                high_count += 1

            family_motif_rows.append(
                {
                    "family": family_name,
                    "motif": int(motif),
                    "n_sequences": len(seq_ids),
                    "mean_jsd": mean_jsd,
                    "median_jsd": median_jsd,
                    "mean_position_span": mean_position_span,
                    "mean_relative_position": family_mean_relpos,
                    "divergence_class": divergence_class,
                    "consensus_relative_position": float("nan"),
                    "consensus_radial_fraction": float("nan"),
                    "consensus_radial_class": "na",
                    "per_sequence_spans": "",
                }
            )

            if divergence_class not in {"low", "medium"}:
                continue

            if representative_runs:
                consensus_relpos = float(np.median(per_seq_mode_positions))
            else:
                consensus_relpos = float("nan")

            radial_values = []
            per_seq_spans = []
            for pdb_id, entity_id in targets:
                seq_id = seq_lookup[(pdb_id, entity_id)]
                runs, run_centers = representative_runs.get(seq_id, ([], []))
                if not runs:
                    continue
                if math.isnan(consensus_relpos):
                    chosen_idx = 0
                else:
                    chosen_idx = min(
                        range(len(runs)),
                        key=lambda idx: abs(run_centers[idx] - consensus_relpos),
                    )
                start, end = runs[chosen_idx]
                per_seq_spans.append(f"{seq_id}:{start}-{end}")

                coord_path = COORD_FILE_OVERRIDES.get(pdb_id)
                if coord_path is None:
                    coord_path = Path(
                        manifest.loc[
                            manifest["pdb_id"] == pdb_id, "assembly_file"
                        ].iloc[0]
                    )
                cache_key = (pdb_id, entity_id)
                if cache_key not in entity_cache:
                    entity_cache[cache_key] = load_entity_coords(
                        coord_path, pdb_id, entity_id
                    )
                coords = entity_cache[cache_key]

                per_residue_radials = []
                residue_names = []
                for seq_pos in range(start, end + 1):
                    residue = coords.label_seq_to_residue.get(seq_pos)
                    radial_pair = coords.radial_stats.get(seq_pos)
                    if residue is not None:
                        residue_names.append(f"{residue.name}{residue.seqid.num}")
                    if radial_pair is not None:
                        per_residue_radials.append(radial_pair[0])
                radial_median = (
                    float(np.median(per_residue_radials))
                    if per_residue_radials
                    else float("nan")
                )
                radial_values.append(radial_median)
                site_rows.append(
                    {
                        "family": family_name,
                        "motif": int(motif),
                        "divergence_class": divergence_class,
                        "pdb_id": pdb_id,
                        "entity_id": entity_id,
                        "seq_id": seq_id,
                        "coord_file": str(coord_path),
                        "rep_chain": coords.rep_chain,
                        "site_start": start,
                        "site_end": end,
                        "site_center_relpos": ((start + end) / 2.0 - 1.0)
                        / int(token_groups[seq_id]["sequence_position"].max()),
                        "radial_fraction": radial_median,
                        "radial_class": radial_class(radial_median),
                        "mapped_residues": ",".join(residue_names),
                    }
                )

            family_motif_rows[-1].update(
                {
                    "consensus_relative_position": consensus_relpos,
                    "consensus_radial_fraction": float(np.nanmedian(radial_values))
                    if radial_values
                    else float("nan"),
                    "consensus_radial_class": radial_class(
                        float(np.nanmedian(radial_values))
                        if radial_values
                        else float("nan")
                    ),
                    "per_sequence_spans": "; ".join(per_seq_spans),
                }
            )

        family_stats = pd.DataFrame(family_motif_rows)
        n_shared = len(family_stats)
        summary_rows.append(
            {
                "family": family_name,
                "n_sequences": len(seq_ids),
                "n_shared_motifs_evaluable": n_shared,
                "jsd_mean_over_motifs": float(family_stats["mean_jsd"].mean())
                if n_shared
                else float("nan"),
                "jsd_median_over_motifs": float(family_stats["median_jsd"].median())
                if n_shared
                else float("nan"),
                "mean_position_span": float(family_stats["mean_position_span"].mean())
                if n_shared
                else float("nan"),
                "frac_low_divergence_jsd_le_0_08": low_count / n_shared
                if n_shared
                else float("nan"),
                "frac_medium_divergence_0_08_0_16": med_count / n_shared
                if n_shared
                else float("nan"),
                "frac_high_divergence_jsd_gt_0_16": high_count / n_shared
                if n_shared
                else float("nan"),
            }
        )
        motif_rows.extend(family_motif_rows)

    summary_df = pd.DataFrame(summary_rows).sort_values("family")
    motif_df = pd.DataFrame(motif_rows).sort_values(
        ["family", "divergence_class", "mean_jsd", "motif"]
    )
    site_df = pd.DataFrame(site_rows).sort_values(
        ["family", "divergence_class", "motif", "pdb_id"]
    )

    summary_df.to_csv(OUTPUT_DIR / "family_summary.csv", index=False)
    motif_df.to_csv(OUTPUT_DIR / "family_shared_motif_stats.csv", index=False)
    site_df.to_csv(OUTPUT_DIR / "family_structural_sites.csv", index=False)

    print(summary_df.to_string(index=False))
    print("\nWrote", OUTPUT_DIR / "family_summary.csv")
    print("Wrote", OUTPUT_DIR / "family_shared_motif_stats.csv")
    print("Wrote", OUTPUT_DIR / "family_structural_sites.csv")


if __name__ == "__main__":
    main()
