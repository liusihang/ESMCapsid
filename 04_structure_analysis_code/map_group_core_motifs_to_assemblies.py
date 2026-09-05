from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from scripts.common.assignment_groups import load_analysis_groups


ROOT = Path("/path/to/science_workspace")
SAE_DIR = ROOT / "ViCapsid" / "SAE"
ASSEMBLY_DIR = ROOT / "pdb_capsid_assemblies"
REPORT_DIR = ASSEMBLY_DIR / "sae_report_20260307"
OUT_DIR = ASSEMBLY_DIR / "group_core_motif_mapping_20260307"
MOTIF_RESULT_DIR = SAE_DIR / "results" / "motif_cluster_conservation" / "ngram_level"

MANIFEST = ASSEMBLY_DIR / "manifest.tsv"
SEQUENCE_REPORT = REPORT_DIR / "final_sequence_predictions.csv"
TOKEN_REPORT = REPORT_DIR / "final_token_semantic_motif.csv"
MAPPING_FILE = SAE_DIR / "motif_id_reindex_mapping.csv"

COORD_FILE_OVERRIDES = {
    "5J7V": ASSEMBLY_DIR / "raw" / "5J7V-entry.cif.gz",
}

N_BINS = 20
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map conserved motif n-grams onto capsid assembly coordinates."
    )
    parser.add_argument("--sae-dir", required=True, type=Path)
    parser.add_argument("--assembly-dir", required=True, type=Path)
    parser.add_argument("--assignment-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coord-override-5j7v", type=Path)
    return parser.parse_args()


@dataclass
class EntityCoords:
    rep_chain: str
    label_seq_to_residue: dict[int, gemmi.Residue]
    radial_fraction_by_label_seq: dict[int, float]


def seq_id_key(seq_id: str) -> tuple[str, int]:
    match = re.match(r"^([0-9A-Za-z]{4})_(\d+)_Chain", seq_id)
    if not match:
        raise ValueError(f"Unsupported Seq_ID: {seq_id}")
    return match.group(1), int(match.group(2))


def radial_class(value: float) -> str:
    if math.isnan(value):
        return "na"
    if value < 0.33:
        return "inner"
    if value < 0.67:
        return "middle"
    return "outer"


def safe_nanmedian(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmedian(arr))


def translate_motif_text(motif_text: str, mapping: dict[int, int]) -> str:
    tokens = [int(token) for token in str(motif_text).split()]
    return " ".join(str(mapping[token]) for token in tokens)


def load_entity_coords(coord_path: Path, entity_id: int) -> EntityCoords:
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
    rep_chain = None
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
            rep_chain = chain.name
            label_seq_to_residue = current
            break
    if rep_chain is None:
        raise KeyError(f"Subchain {rep_subchain} not found in {coord_path}")

    all_points = []
    rep_points = []
    for chain in model:
        for residue in chain:
            if residue.het_flag != "A":
                continue
            atom = residue.find_atom("CA", "\0")
            if atom is None:
                continue
            point = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            all_points.append(point)
            label_seq = getattr(residue, "label_seq", None)
            if chain.name == rep_chain and label_seq is not None:
                rep_points.append((int(label_seq), point))

    if all_points:
        center = np.mean(np.vstack(all_points), axis=0)
        distances = [float(np.linalg.norm(point - center)) for point in all_points]
        min_dist = min(distances)
        max_dist = max(distances)
        denom = max(max_dist - min_dist, 1e-6)
    else:
        center = np.zeros(3, dtype=float)
        min_dist = 0.0
        denom = 1.0

    radial_fraction_by_label_seq = {}
    for label_seq, point in rep_points:
        dist = float(np.linalg.norm(point - center))
        radial_fraction_by_label_seq[label_seq] = (dist - min_dist) / denom

    return EntityCoords(
        rep_chain=rep_chain,
        label_seq_to_residue=label_seq_to_residue,
        radial_fraction_by_label_seq=radial_fraction_by_label_seq,
    )


def find_ngram_hits(
    labels: np.ndarray, motif_tokens: list[int]
) -> list[tuple[int, int]]:
    motif_len = len(motif_tokens)
    if motif_len == 0 or len(labels) < motif_len:
        return []
    hits = []
    for start in range(len(labels) - motif_len + 1):
        if np.array_equal(labels[start : start + motif_len], motif_tokens):
            hits.append((start, start + motif_len - 1))
    return hits


def main() -> None:
    global SAE_DIR, ASSEMBLY_DIR, REPORT_DIR, OUT_DIR, MOTIF_RESULT_DIR
    global MANIFEST, SEQUENCE_REPORT, TOKEN_REPORT, MAPPING_FILE
    global COORD_FILE_OVERRIDES

    args = parse_args()
    SAE_DIR = args.sae_dir.expanduser().resolve()
    ASSEMBLY_DIR = args.assembly_dir.expanduser().resolve()
    REPORT_DIR = ASSEMBLY_DIR / "sae_report_20260307"
    OUT_DIR = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else ASSEMBLY_DIR / "group_core_motif_mapping_20260307"
    )
    MOTIF_RESULT_DIR = (
        SAE_DIR / "results" / "motif_cluster_conservation" / "ngram_level"
    )
    MANIFEST = ASSEMBLY_DIR / "manifest.tsv"
    SEQUENCE_REPORT = REPORT_DIR / "final_sequence_predictions.csv"
    TOKEN_REPORT = REPORT_DIR / "final_token_semantic_motif.csv"
    MAPPING_FILE = SAE_DIR / "motif_id_reindex_mapping.csv"
    COORD_FILE_OVERRIDES = {}
    if args.coord_override_5j7v:
        COORD_FILE_OVERRIDES["5J7V"] = args.coord_override_5j7v.expanduser().resolve()

    groups = load_analysis_groups(args.assignment_csv.expanduser().resolve())
    core_files = {
        group: MOTIF_RESULT_DIR / f"{group}_motif_stats.csv" for group in groups
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, sep="\t")
    seq_df = pd.read_csv(SEQUENCE_REPORT)
    token_df = pd.read_csv(TOKEN_REPORT)
    mapping_df = pd.read_csv(MAPPING_FILE)
    reindexed_to_original = dict(
        zip(mapping_df["reindexed_id"], mapping_df["original_id"])
    )

    seq_df["key"] = seq_df["Seq_ID"].map(seq_id_key)
    token_groups = {
        seq_id: group.sort_values("token_idx_in_sequence").reset_index(drop=True)
        for seq_id, group in token_df.groupby("Seq_ID")
    }

    coord_cache: dict[tuple[str, int], EntityCoords] = {}
    summary_rows = []
    motif_rows = []
    hit_rows = []
    bin_rows = []

    for group_name, cluster_ids in groups.items():
        core_df = pd.read_csv(core_files[group_name])
        core_motifs_reindexed = (
            core_df.loc[core_df["is_core"], "motif"].astype(str).tolist()
        )
        core_motifs_original = [
            translate_motif_text(motif, reindexed_to_original)
            for motif in core_motifs_reindexed
        ]

        structure_seq_df = seq_df[seq_df["Pred_Label"].isin(cluster_ids)].copy()
        structure_seq_df = structure_seq_df.sort_values(["Seq_ID"]).reset_index(
            drop=True
        )
        seq_ids = structure_seq_df["Seq_ID"].tolist()

        per_seq_coverage = {}
        group_hit_counter = Counter()
        group_unique_motifs = set()

        for seq_id in seq_ids:
            seq_tokens = token_groups[seq_id]
            seq_len = int(seq_tokens["sequence_position"].max())
            per_seq_coverage[seq_id] = np.zeros(seq_len, dtype=int)

        for motif_text_reindexed, motif_text_original in zip(
            core_motifs_reindexed, core_motifs_original
        ):
            motif_tokens = [int(token) for token in motif_text_original.split()]
            motif_len = len(motif_tokens)
            motif_hit_count = 0
            motif_hit_structures = set()
            motif_centers = []
            motif_radials = []

            for _, seq_row in structure_seq_df.iterrows():
                seq_id = seq_row["Seq_ID"]
                pdb_id, entity_id = seq_row["key"]
                seq_tokens = token_groups[seq_id]
                labels = seq_tokens["semantic_motif_label"].astype(int).to_numpy()
                seq_positions = seq_tokens["sequence_position"].astype(int).to_numpy()
                seq_len = int(seq_positions.max())
                hits = find_ngram_hits(labels, motif_tokens)
                if not hits:
                    continue

                motif_hit_structures.add(seq_id)
                group_unique_motifs.add(motif_text_reindexed)
                motif_hit_count += len(hits)
                group_hit_counter[motif_text_reindexed] += len(hits)

                coord_path = COORD_FILE_OVERRIDES.get(pdb_id)
                if coord_path is None:
                    coord_path = Path(
                        manifest.loc[
                            manifest["pdb_id"] == pdb_id, "assembly_file"
                        ].iloc[0]
                    )
                cache_key = (pdb_id, entity_id)
                if cache_key not in coord_cache:
                    coord_cache[cache_key] = load_entity_coords(coord_path, entity_id)
                coords = coord_cache[cache_key]

                for start_idx, end_idx in hits:
                    start_pos = int(seq_positions[start_idx])
                    end_pos = int(seq_positions[end_idx])
                    per_seq_coverage[seq_id][start_pos - 1 : end_pos] += 1

                    rel_start = (start_pos - 1) / seq_len
                    rel_center = ((start_pos + end_pos) / 2.0 - 1.0) / seq_len
                    motif_centers.append(rel_center)

                    residue_names = []
                    radial_values = []
                    for seq_pos in range(start_pos, end_pos + 1):
                        residue = coords.label_seq_to_residue.get(seq_pos)
                        radial_value = coords.radial_fraction_by_label_seq.get(
                            seq_pos, float("nan")
                        )
                        if residue is not None:
                            residue_names.append(f"{residue.name}{residue.seqid.num}")
                        if not math.isnan(radial_value):
                            radial_values.append(radial_value)
                    radial_fraction = (
                        float(np.median(radial_values))
                        if radial_values
                        else float("nan")
                    )
                    motif_radials.append(radial_fraction)

                    hit_rows.append(
                        {
                            "group": group_name,
                            "reindexed_motif": motif_text_reindexed,
                            "original_motif": motif_text_original,
                            "motif_len": motif_len,
                            "seq_id": seq_id,
                            "pdb_id": pdb_id,
                            "entity_id": entity_id,
                            "pred_label": int(seq_row["Pred_Label"]),
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                            "rel_start": rel_start,
                            "rel_center": rel_center,
                            "coord_file": str(coord_path),
                            "rep_chain": coords.rep_chain,
                            "radial_fraction": radial_fraction,
                            "radial_class": radial_class(radial_fraction),
                            "mapped_residues": ",".join(residue_names),
                        }
                    )

            motif_rows.append(
                {
                    "group": group_name,
                    "reindexed_motif": motif_text_reindexed,
                    "original_motif": motif_text_original,
                    "motif_len": motif_len,
                    "is_core": True,
                    "observed_in_structure_set": motif_hit_count > 0,
                    "n_structures_with_hit": len(motif_hit_structures),
                    "n_total_hits": motif_hit_count,
                    "mean_rel_center": float(np.mean(motif_centers))
                    if motif_centers
                    else float("nan"),
                    "position_span": float(max(motif_centers) - min(motif_centers))
                    if motif_centers
                    else float("nan"),
                    "median_radial_fraction": safe_nanmedian(motif_radials),
                    "radial_class": radial_class(safe_nanmedian(motif_radials)),
                }
            )

        total_core_hits = 0
        radial_counter = Counter()
        for seq_id, coverage in per_seq_coverage.items():
            total_core_hits += int(coverage.sum())
            pdb_id, entity_id = seq_id_key(seq_id)
            coord_path = COORD_FILE_OVERRIDES.get(pdb_id)
            if coord_path is None:
                coord_path = Path(
                    manifest.loc[manifest["pdb_id"] == pdb_id, "assembly_file"].iloc[0]
                )
            cache_key = (pdb_id, entity_id)
            if cache_key not in coord_cache:
                coord_cache[cache_key] = load_entity_coords(coord_path, entity_id)
            coords = coord_cache[cache_key]

            for residue_index, cov in enumerate(coverage, start=1):
                if cov == 0:
                    continue
                radial_value = coords.radial_fraction_by_label_seq.get(
                    residue_index, float("nan")
                )
                radial_counter[radial_class(radial_value)] += cov

            seq_len = len(coverage)
            bin_idx = np.minimum(
                (np.arange(seq_len) / seq_len * N_BINS).astype(int), N_BINS - 1
            )
            seq_bin_df = pd.DataFrame({"bin": bin_idx, "coverage": coverage})
            for bin_id, cov_sum in seq_bin_df.groupby("bin")["coverage"].sum().items():
                bin_rows.append(
                    {
                        "group": group_name,
                        "seq_id": seq_id,
                        "bin": int(bin_id),
                        "bin_left": BIN_EDGES[int(bin_id)],
                        "bin_right": BIN_EDGES[int(bin_id) + 1],
                        "coverage_sum": int(cov_sum),
                    }
                )

        group_bins = pd.DataFrame(
            [row for row in bin_rows if row["group"] == group_name]
        )
        if len(group_bins):
            group_bin_cov = group_bins.groupby(
                ["bin", "bin_left", "bin_right"], as_index=False
            )["coverage_sum"].sum()
            group_bin_cov["coverage_frac"] = group_bin_cov["coverage_sum"] / max(
                group_bin_cov["coverage_sum"].sum(), 1
            )
            top_bins = group_bin_cov.sort_values(
                ["coverage_frac", "bin"], ascending=[False, True]
            ).head(5)
            top_bin_label = "; ".join(
                f"{row.bin_left:.2f}-{row.bin_right:.2f}:{row.coverage_frac:.3f}"
                for _, row in top_bins.iterrows()
            )
        else:
            top_bin_label = ""

        summary_rows.append(
            {
                "group": group_name,
                "n_core_motifs": len(core_motifs_reindexed),
                "n_structure_sequences": len(seq_ids),
                "n_core_motifs_observed_in_structures": len(group_unique_motifs),
                "frac_core_motifs_observed": len(group_unique_motifs)
                / max(len(core_motifs_reindexed), 1),
                "total_core_hit_spans": sum(group_hit_counter.values()),
                "radial_frac_inner": radial_counter["inner"] / max(total_core_hits, 1),
                "radial_frac_middle": radial_counter["middle"]
                / max(total_core_hits, 1),
                "radial_frac_outer": radial_counter["outer"] / max(total_core_hits, 1),
                "top_coverage_bins": top_bin_label,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("group")
    motif_df = pd.DataFrame(motif_rows).sort_values(
        [
            "group",
            "observed_in_structure_set",
            "n_structures_with_hit",
            "n_total_hits",
            "reindexed_motif",
        ],
        ascending=[True, False, False, False, True],
    )
    hit_df = pd.DataFrame(hit_rows).sort_values(
        ["group", "reindexed_motif", "seq_id", "start_pos"]
    )
    bin_df = pd.DataFrame(bin_rows).sort_values(["group", "seq_id", "bin"])

    summary_df.to_csv(OUT_DIR / "group_core_mapping_summary.csv", index=False)
    motif_df.to_csv(OUT_DIR / "group_core_mapping_per_motif.csv", index=False)
    hit_df.to_csv(OUT_DIR / "group_core_mapping_hits.csv", index=False)
    bin_df.to_csv(OUT_DIR / "group_core_mapping_bins.csv", index=False)

    print(summary_df.to_string(index=False))
    print("\nWrote", OUT_DIR / "group_core_mapping_summary.csv")
    print("Wrote", OUT_DIR / "group_core_mapping_per_motif.csv")
    print("Wrote", OUT_DIR / "group_core_mapping_hits.csv")
    print("Wrote", OUT_DIR / "group_core_mapping_bins.csv")


if __name__ == "__main__":
    main()
