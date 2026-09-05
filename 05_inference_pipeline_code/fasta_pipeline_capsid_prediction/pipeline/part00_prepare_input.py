from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    PipelineError,
    chunk_records,
    dedupe_seq_ids,
    ensure_dir,
    is_complete,
    normalize_seq,
    parse_fasta,
    safe_seq_id,
    save_json,
    save_part_summary,
    write_csv,
    write_fasta,
)


def run(
    run_dir: Path, fasta_path: Path, config: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    manifest_dir = ensure_dir(run_dir / "manifest")
    summary_path = manifest_dir / "prepare.summary.json"
    chunk_manifest_path = manifest_dir / "chunk_manifest.csv"
    run_manifest_path = manifest_dir / "run_manifest.json"
    global_manifest_path = manifest_dir / "sequence_manifest.all.csv"
    dropped_manifest_path = manifest_dir / "dropped_sequences.csv"

    if not force and is_complete(
        summary_path,
        [chunk_manifest_path, run_manifest_path, global_manifest_path],
    ):
        return {"status": "skipped", "summary_path": str(summary_path)}

    records = parse_fasta(fasta_path)
    normalized: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        cleaned = normalize_seq(record["sequence"])
        if not cleaned:
            dropped.append(
                {
                    "record_index": index,
                    "Original_Header": record["header"],
                    "drop_reason": "empty_sequence_after_cleaning",
                }
            )
            continue
        normalized.append(
            {
                "record_index": index,
                "Seq_ID": safe_seq_id(record["header"], index),
                "Original_Header": record["header"],
                "seq": cleaned,
                "aa_len": len(cleaned),
            }
        )

    normalized = dedupe_seq_ids(normalized)
    if not normalized:
        raise PipelineError("All FASTA records were dropped during prepare.")

    pipeline_cfg = config.get("pipeline", {})
    chunks = chunk_records(
        normalized,
        int(pipeline_cfg.get("max_sequences_per_chunk", 2000)),
        int(pipeline_cfg.get("max_total_aa_per_chunk", 800000)),
    )

    all_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk_id = f"chunk_{chunk_index:05d}"
        chunk_dir = ensure_dir(run_dir / "chunks" / chunk_id / "00_input")
        chunk_fasta_path = chunk_dir / "input.normalized.fasta"
        chunk_manifest = chunk_dir / "sequence_manifest.csv"
        chunk_summary_path = chunk_dir / "prepare.summary.json"

        materialized = []
        total_aa = 0
        for seq_index_in_chunk, row in enumerate(chunk, start=1):
            updated = {
                **row,
                "chunk_id": chunk_id,
                "sequence_index_in_chunk": seq_index_in_chunk,
            }
            materialized.append(updated)
            all_rows.append(updated)
            total_aa += int(updated["aa_len"])

        write_fasta(chunk_fasta_path, materialized)
        write_csv(
            chunk_manifest,
            materialized,
            [
                "record_index",
                "Seq_ID",
                "Seq_ID_Base",
                "Original_Header",
                "seq",
                "aa_len",
                "chunk_id",
                "sequence_index_in_chunk",
            ],
        )
        save_part_summary(
            chunk_summary_path,
            "prepare",
            {
                "chunk_id": chunk_id,
                "sequence_count": len(materialized),
                "total_aa": total_aa,
                "outputs": {
                    "input_fasta": str(chunk_fasta_path),
                    "sequence_manifest": str(chunk_manifest),
                },
            },
        )
        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "sequence_count": len(materialized),
                "total_aa": total_aa,
                "input_fasta": str(chunk_fasta_path),
                "sequence_manifest": str(chunk_manifest),
            }
        )

    write_csv(
        global_manifest_path,
        all_rows,
        [
            "record_index",
            "Seq_ID",
            "Seq_ID_Base",
            "Original_Header",
            "seq",
            "aa_len",
            "chunk_id",
            "sequence_index_in_chunk",
        ],
    )
    write_csv(
        chunk_manifest_path,
        chunk_rows,
        ["chunk_id", "sequence_count", "total_aa", "input_fasta", "sequence_manifest"],
    )
    if dropped:
        write_csv(
            dropped_manifest_path,
            dropped,
            ["record_index", "Original_Header", "drop_reason"],
        )
    else:
        dropped_manifest_path.write_text(
            "record_index,Original_Header,drop_reason\n", encoding="utf-8"
        )

    run_manifest = {
        "source_fasta": str(fasta_path),
        "sequence_count": len(all_rows),
        "dropped_count": len(dropped),
        "chunk_count": len(chunk_rows),
        "chunk_ids": [row["chunk_id"] for row in chunk_rows],
    }
    save_json(run_manifest_path, run_manifest)
    save_part_summary(
        summary_path,
        "prepare",
        {
            **run_manifest,
            "outputs": {
                "sequence_manifest_all": str(global_manifest_path),
                "chunk_manifest": str(chunk_manifest_path),
                "run_manifest": str(run_manifest_path),
                "dropped_manifest": str(dropped_manifest_path),
            },
        },
    )
    return {"status": "completed", "summary_path": str(summary_path)}
