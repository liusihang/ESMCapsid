from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .common import (
    PipelineError,
    ensure_dir,
    read_csv_rows,
    resolve_chunk_ids,
    save_part_summary,
    write_csv,
    write_parquet_rows,
)


def _merge_sequence_predictions(
    prediction_rows: list[dict[str, str]],
    motif_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    motif_by_seq = {row["Seq_ID"]: row for row in motif_rows}
    merged: list[dict[str, Any]] = []
    for row in prediction_rows:
        seq_id = row["Seq_ID"]
        motif = motif_by_seq.get(seq_id, {})
        merged.append(
            {
                "Seq_ID": seq_id,
                "Pred_Label": row.get("Pred_Label", ""),
                "Pred_Confidence": row.get("Confidence", ""),
                "Top_Semantic_Motif": motif.get("Top_Semantic_Motif", ""),
                "Top_Semantic_Motif_Fraction": motif.get(
                    "Top_Semantic_Motif_Fraction", ""
                ),
                "Top_Semantic_Motif_Weight": motif.get("Top_Semantic_Motif_Weight", ""),
                "Mapped_Tokens": motif.get("Mapped_Tokens", ""),
                "Total_Tokens": motif.get("Total_Tokens", ""),
                "Unique_Semantic_Motifs": motif.get("Unique_Semantic_Motifs", ""),
                "chunk_id": row.get("chunk_id", ""),
            }
        )
    return merged


def run(run_dir: Path, config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    report_dir = ensure_dir(run_dir / "60_report")
    summary_path = report_dir / "report.summary.json"
    pipeline_report_path = report_dir / "pipeline_report.json"
    final_sequence_predictions = report_dir / "final_sequence_predictions.csv"
    final_token_table = report_dir / "final_token_semantic_motif.parquet"
    if (
        not force
        and summary_path.exists()
        and pipeline_report_path.exists()
        and final_sequence_predictions.exists()
        and final_token_table.exists()
    ):
        return {"status": "skipped", "summary_path": str(summary_path)}

    chunk_ids = resolve_chunk_ids(run_dir)
    prediction_rows: list[dict[str, str]] = []
    motif_rows: list[dict[str, str]] = []
    token_rows: list[dict[str, str]] = []
    completed_chunks = []
    for chunk_id in chunk_ids:
        chunk_dir = run_dir / "chunks" / chunk_id
        pred_path = chunk_dir / "50_seq_pred" / "sequence_knn_predictions.csv"
        motif_path = (
            chunk_dir / "30_semantic_motif" / "sequence_semantic_motif_summary.csv"
        )
        token_path = chunk_dir / "30_semantic_motif" / "token_semantic_motif_table.csv"
        if not pred_path.exists() or not motif_path.exists() or not token_path.exists():
            raise PipelineError(
                f"Report inputs missing for {chunk_id}: {pred_path}, {motif_path}, {token_path}"
            )
        for row in read_csv_rows(pred_path):
            updated = dict(row)
            updated["chunk_id"] = chunk_id
            prediction_rows.append(updated)
        for row in read_csv_rows(motif_path):
            updated = dict(row)
            updated["chunk_id"] = chunk_id
            motif_rows.append(updated)
        for row in read_csv_rows(token_path):
            updated = dict(row)
            updated["chunk_id"] = chunk_id
            token_rows.append(updated)
        completed_chunks.append(chunk_id)

    merged_rows = _merge_sequence_predictions(prediction_rows, motif_rows)
    write_csv(
        final_sequence_predictions,
        merged_rows,
        [
            "Seq_ID",
            "Pred_Label",
            "Pred_Confidence",
            "Top_Semantic_Motif",
            "Top_Semantic_Motif_Fraction",
            "Top_Semantic_Motif_Weight",
            "Mapped_Tokens",
            "Total_Tokens",
            "Unique_Semantic_Motifs",
            "chunk_id",
        ],
    )
    write_parquet_rows(
        final_token_table,
        token_rows,
        [
            "Seq_ID",
            "token_idx_global",
            "token_idx_in_sequence",
            "sequence_position",
            "aa",
            "semantic_motif_label",
            "semantic_motif_confidence",
            "chunk_id",
        ],
    )

    payload = {
        "chunk_count": len(completed_chunks),
        "sequence_count": len(merged_rows),
        "token_annotation_count": len(token_rows),
        "outputs": {
            "final_sequence_predictions": str(final_sequence_predictions),
            "final_token_semantic_motif": str(final_token_table),
            "pipeline_report": str(pipeline_report_path),
        },
    }
    save_part_summary(summary_path, "report", payload)
    save_part_summary(pipeline_report_path, "report", payload)
    return {"status": "completed", "summary_path": str(summary_path)}
