from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ChunkContext,
    PipelineError,
    canonical_python_executable,
    ensure_dir,
    is_complete,
    load_json,
    read_csv_rows,
    run_command,
    save_part_summary,
    write_csv,
)


def _build_token_table(
    manifest_rows: list[dict[str, str]],
    labels: np.ndarray,
    confidence: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_rows: list[dict[str, Any]] = []
    sequence_scores: dict[str, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    sequence_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_tokens": 0, "mapped_tokens": 0, "unique_motifs": 0}
    )

    for row in manifest_rows:
        seq_id = row.get("prot_id") or row.get("Seq_ID")
        sequence = row["seq"]
        tok_start = int(row["tok_start"])
        tok_end = int(row["tok_end"])
        local_labels = labels[tok_start:tok_end]
        local_conf = confidence[tok_start:tok_end]
        motif_set = set()
        for token_global_idx, (label, conf) in enumerate(
            zip(local_labels, local_conf),
            start=tok_start,
        ):
            seq_pos = token_global_idx - tok_start + 1
            aa = sequence[seq_pos - 1] if seq_pos - 1 < len(sequence) else "X"
            token_rows.append(
                {
                    "Seq_ID": seq_id,
                    "token_idx_global": token_global_idx,
                    "token_idx_in_sequence": seq_pos - 1,
                    "sequence_position": seq_pos,
                    "aa": aa,
                    "semantic_motif_label": int(label),
                    "semantic_motif_confidence": float(conf),
                }
            )
            sequence_stats[seq_id]["total_tokens"] += 1
            if int(label) >= 0:
                sequence_stats[seq_id]["mapped_tokens"] += 1
                sequence_scores[seq_id][int(label)] += float(conf)
                motif_set.add(int(label))
        sequence_stats[seq_id]["unique_motifs"] = len(motif_set)

    sequence_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        seq_id = row.get("prot_id") or row.get("Seq_ID")
        scores = sequence_scores[seq_id]
        total_weight = float(sum(scores.values()))
        if scores:
            top_label, top_weight = max(scores.items(), key=lambda item: item[1])
            top_fraction = float(top_weight / total_weight) if total_weight > 0 else 0.0
        else:
            top_label, top_weight, top_fraction = -1, 0.0, 0.0
        sequence_rows.append(
            {
                "Seq_ID": seq_id,
                "Top_Semantic_Motif": int(top_label),
                "Top_Semantic_Motif_Weight": float(top_weight),
                "Top_Semantic_Motif_Fraction": float(top_fraction),
                "Total_Tokens": int(sequence_stats[seq_id]["total_tokens"]),
                "Mapped_Tokens": int(sequence_stats[seq_id]["mapped_tokens"]),
                "Unique_Semantic_Motifs": int(sequence_stats[seq_id]["unique_motifs"]),
            }
        )
    return token_rows, sequence_rows


def run(
    ctx: ChunkContext,
    config: dict[str, Any],
    map_backend: str = "gpu",
    force: bool = False,
) -> dict[str, Any]:
    stage_dir = ensure_dir(ctx.stage_dir("map_semantic_motif"))
    summary_path = stage_dir / "semantic_motif.summary.json"
    required_outputs = [
        stage_dir / "token_semantic_motif_labels.npy",
        stage_dir / "token_semantic_motif_confidence.npy",
        stage_dir / "token_semantic_motif_table.csv",
        stage_dir / "sequence_semantic_motif_summary.csv",
    ]
    if not force and is_complete(summary_path, required_outputs):
        return {"status": "skipped", "summary_path": str(summary_path)}

    sae_dir = ctx.stage_dir("sae")
    query_npz = sae_dir / "sae_token_features.npz"
    if not query_npz.exists():
        raise PipelineError(f"SAE token features missing: {query_npz}")

    backend_cfg = config["backends"]["map_semantic_motif"]
    reference_cfg = config["reference"]
    backend_output = ensure_dir(stage_dir / "backend_output")
    output_prefix = backend_output / "token_semantic_motif"
    command = [
        canonical_python_executable(config, backend_cfg),
        str(Path(backend_cfg["script_path"])),
        "--query_npz",
        str(query_npz),
        "--ref_npz",
        str(reference_cfg["semantic_motif_ref_npz"]),
        "--ref_labels",
        str(reference_cfg["semantic_motif_labels"]),
        "--trained_index",
        str(reference_cfg["semantic_motif_index"]),
        "--output_prefix",
        str(output_prefix),
        "--k",
        str(backend_cfg.get("k", 20)),
        "--nprobe",
        str(backend_cfg.get("nprobe", 64)),
        "--add_batch_size",
        str(backend_cfg.get("add_batch_size", 200000)),
        "--query_batch_size",
        str(backend_cfg.get("query_batch_size", 50000)),
    ]
    if map_backend == "cpu":
        command.extend(["--gpu", "-2"])
    else:
        command.extend(["--gpu", str(backend_cfg.get("gpu_id", 0))])
    if backend_cfg.get("use_float16", True) and map_backend != "cpu":
        command.append("--use_float16")

    run_command(command, stage_dir, summary_path.name)

    raw_labels = output_prefix.with_name(output_prefix.name + "_clusters.npy")
    raw_conf = output_prefix.with_name(output_prefix.name + "_confidence.npy")
    raw_summary = output_prefix.with_name(output_prefix.name + "_summary.json")
    if not raw_labels.exists() or not raw_conf.exists() or not raw_summary.exists():
        raise PipelineError(
            f"Semantic motif backend outputs missing: {raw_labels}, {raw_conf}, {raw_summary}"
        )

    labels = np.load(raw_labels)
    confidence = np.load(raw_conf)
    shutil.copy2(raw_labels, stage_dir / "token_semantic_motif_labels.npy")
    shutil.copy2(raw_conf, stage_dir / "token_semantic_motif_confidence.npy")

    manifest_rows = read_csv_rows(
        ctx.stage_dir("embed") / "sequence_manifest.filtered.csv"
    )
    token_rows, sequence_rows = _build_token_table(manifest_rows, labels, confidence)
    write_csv(
        stage_dir / "token_semantic_motif_table.csv",
        token_rows,
        [
            "Seq_ID",
            "token_idx_global",
            "token_idx_in_sequence",
            "sequence_position",
            "aa",
            "semantic_motif_label",
            "semantic_motif_confidence",
        ],
    )
    write_csv(
        stage_dir / "sequence_semantic_motif_summary.csv",
        sequence_rows,
        [
            "Seq_ID",
            "Top_Semantic_Motif",
            "Top_Semantic_Motif_Weight",
            "Top_Semantic_Motif_Fraction",
            "Total_Tokens",
            "Mapped_Tokens",
            "Unique_Semantic_Motifs",
        ],
    )
    backend_summary = load_json(raw_summary)
    save_part_summary(
        summary_path,
        "map_semantic_motif",
        {
            "chunk_id": ctx.chunk_id,
            "map_backend": map_backend,
            "backend_summary": backend_summary,
            "outputs": {
                "token_labels": str(stage_dir / "token_semantic_motif_labels.npy"),
                "token_confidence": str(
                    stage_dir / "token_semantic_motif_confidence.npy"
                ),
                "token_table": str(stage_dir / "token_semantic_motif_table.csv"),
                "sequence_summary": str(
                    stage_dir / "sequence_semantic_motif_summary.csv"
                ),
            },
        },
    )
    return {"status": "completed", "summary_path": str(summary_path)}
