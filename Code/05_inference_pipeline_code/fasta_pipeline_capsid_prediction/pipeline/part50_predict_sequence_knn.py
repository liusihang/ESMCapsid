from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    ChunkContext,
    PipelineError,
    canonical_python_executable,
    ensure_dir,
    is_complete,
    run_command,
    save_part_summary,
)


def run(
    ctx: ChunkContext, config: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    stage_dir = ensure_dir(ctx.stage_dir("predict_seq"))
    part_summary_path = stage_dir / "predict_seq.summary.json"
    backend_summary_path = stage_dir / "sequence_knn_summary.json"
    required_outputs = [
        stage_dir / "sequence_knn_predictions.csv",
        backend_summary_path,
    ]
    if (
        not force
        and part_summary_path.exists()
        and all(path.exists() for path in required_outputs)
    ):
        return {"status": "skipped", "summary_path": str(part_summary_path)}

    pool_dir = ctx.stage_dir("pool_seq")
    query_npz = pool_dir / "sequence_sae_mean_features.npz"
    if not query_npz.exists():
        raise PipelineError(f"Sequence mean features missing: {query_npz}")

    embed_manifest = ctx.stage_dir("embed") / "sequence_manifest.filtered.csv"
    backend_cfg = config["backends"]["predict_seq"]
    reference_cfg = config["reference"]
    command = [
        canonical_python_executable(config, backend_cfg),
        str(Path(backend_cfg["script_path"])),
        "--ref_npz",
        str(reference_cfg["sequence_knn_ref_npz"]),
        "--ref_labels_csv",
        str(reference_cfg["sequence_knn_ref_labels_csv"]),
        "--ref_label_col",
        str(reference_cfg.get("sequence_knn_label_col", "Cluster")),
        "--query_npz",
        str(query_npz),
        "--query_ids_csv",
        str(embed_manifest),
        "--query_id_col",
        str(backend_cfg.get("query_id_col", "prot_id")),
        "--out_dir",
        str(stage_dir),
        "--prefix",
        "sequence_knn",
        "--k",
        str(backend_cfg.get("k", 20)),
        "--backend",
        str(backend_cfg.get("backend", "auto")),
        "--query_batch",
        str(backend_cfg.get("query_batch", 5000)),
        "--cpu_threads",
        str(backend_cfg.get("cpu_threads", 7)),
    ]
    if backend_cfg.get("include_noise_in_vote", True):
        command.append("--include_noise_in_vote")
    run_command(command, stage_dir, part_summary_path.name)
    if not (stage_dir / "sequence_knn_predictions.csv").exists():
        raise PipelineError("kNN prediction output missing after backend run.")
    save_part_summary(
        part_summary_path,
        "predict_seq",
        {
            "chunk_id": ctx.chunk_id,
            "outputs": {
                "sequence_knn_predictions": str(
                    stage_dir / "sequence_knn_predictions.csv"
                ),
                "sequence_knn_summary": str(backend_summary_path),
            },
        },
    )
    return {"status": "completed", "summary_path": str(part_summary_path)}
