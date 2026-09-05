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
    stage_dir = ensure_dir(ctx.stage_dir("pool_seq"))
    summary_path = stage_dir / "pool_seq.summary.json"
    output_path = stage_dir / "sequence_sae_mean_features.npz"
    if not force and is_complete(summary_path, [output_path]):
        return {"status": "skipped", "summary_path": str(summary_path)}

    sae_dir = ctx.stage_dir("sae")
    token_features = sae_dir / "sae_token_features.npz"
    token_map = sae_dir / "token_to_sequence_map.npz"
    if not token_features.exists() or not token_map.exists():
        raise PipelineError(f"Missing SAE inputs: {token_features}, {token_map}")

    backend_cfg = config["backends"]["pool_seq"]
    command = [
        canonical_python_executable(config, backend_cfg),
        str(Path(backend_cfg["script_path"])),
        "--token_npz",
        str(token_features),
        "--mapping_npz",
        str(token_map),
        "--out_npz",
        str(output_path),
    ]
    run_command(command, stage_dir, summary_path.name)
    save_part_summary(
        summary_path,
        "pool_seq",
        {
            "chunk_id": ctx.chunk_id,
            "outputs": {"sequence_sae_mean_features": str(output_path)},
        },
    )
    return {"status": "completed", "summary_path": str(summary_path)}
