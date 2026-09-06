from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import (
    ChunkContext,
    PipelineError,
    canonical_python_executable,
    copy_or_symlink,
    ensure_dir,
    is_complete,
    load_json,
    run_command,
    save_json,
    save_part_summary,
)


def _select_token_file(stage_dir: Path, token_file_name: str) -> Path:
    candidate = stage_dir / "token_layers" / token_file_name
    if candidate.exists():
        return candidate
    matches = sorted((stage_dir / "token_layers").glob("*.tokens.npy"))
    if len(matches) == 1:
        return matches[0]
    raise PipelineError(
        f"Token input file not found: {candidate}. Available: {[path.name for path in matches]}"
    )


def run(
    ctx: ChunkContext, config: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    stage_dir = ensure_dir(ctx.stage_dir("sae"))
    summary_path = stage_dir / "sae.summary.json"
    required_outputs = [
        stage_dir / "sae_token_features.npz",
        stage_dir / "token_to_sequence_map.npz",
    ]
    if not force and is_complete(summary_path, required_outputs):
        return {"status": "skipped", "summary_path": str(summary_path)}

    embed_dir = ctx.stage_dir("embed")
    filtered_manifest = embed_dir / "sequence_manifest.filtered.csv"
    if not filtered_manifest.exists():
        raise PipelineError(f"Filtered manifest missing: {filtered_manifest}")

    backend_cfg = config["backends"]["sae"]
    token_file = _select_token_file(
        embed_dir, str(backend_cfg.get("token_file_name", "layer_28.tokens.npy"))
    )

    cfg_dir = ensure_dir(stage_dir / "infer_cfg")
    backend_root = ensure_dir(stage_dir / "backend_output")
    dims = int(backend_cfg["dims"])
    ks = int(backend_cfg["ks"])
    backend_run_dir = ensure_dir(backend_root / f"SAE_Tok_D{dims}_K{ks}")
    copy_or_symlink(
        Path(backend_cfg["pretrained_model_path"]), backend_run_dir / "sae_model.pt"
    )
    copy_or_symlink(
        Path(backend_cfg["global_stats_path"]), backend_root / "global_stats.npz"
    )

    preprocess_config = {
        "input_tokens_path": str(token_file),
        "input_meta_path": str(filtered_manifest),
        "d_in": int(backend_cfg["input_hidden_dim"]),
        "n_epochs": 1,
        "steps_per_epoch": 1,
        "batch_size": int(backend_cfg.get("infer_chunk_size", 4096)),
        "epoch_files": [],
    }
    infer_config = {
        "output_root": str(backend_root),
        "sweep_dims": [dims],
        "sweep_ks": [ks],
        "use_local_staging": False,
        "infer_chunk_size": int(backend_cfg.get("infer_chunk_size", 4096)),
        "use_amp": bool(backend_cfg.get("use_amp", True)),
        "use_torch_compile": bool(backend_cfg.get("use_torch_compile", False)),
        "do_mean_center": bool(backend_cfg.get("do_mean_center", True)),
        "do_rescale": bool(backend_cfg.get("do_rescale", True)),
        "clip_extreme_values": bool(backend_cfg.get("clip_extreme_values", True)),
    }
    preprocess_config_path = cfg_dir / "preprocess_config.json"
    infer_config_path = cfg_dir / "infer_config.json"
    save_json(preprocess_config_path, preprocess_config)
    save_json(infer_config_path, infer_config)

    command = [
        canonical_python_executable(config, backend_cfg),
        str(Path(backend_cfg["script_path"])),
        "--data_dir",
        str(cfg_dir),
        "--config",
        str(infer_config_path),
        "--stage",
        "infer_only",
        "--dims",
        str(dims),
        "--ks",
        str(ks),
        "--output_root",
        str(backend_root),
    ]
    run_command(command, stage_dir, summary_path.name)

    backend_token_features = backend_run_dir / "sae_feats_token_csr.npz"
    backend_token_map = backend_run_dir / "token_to_seq_mapping.npz"
    if not backend_token_features.exists() or not backend_token_map.exists():
        raise PipelineError(
            f"SAE backend outputs missing: {backend_token_features}, {backend_token_map}"
        )

    shutil.copy2(backend_token_features, stage_dir / "sae_token_features.npz")
    shutil.copy2(backend_token_map, stage_dir / "token_to_sequence_map.npz")
    save_part_summary(
        summary_path,
        "sae",
        {
            "chunk_id": ctx.chunk_id,
            "backend_output_root": str(backend_root),
            "outputs": {
                "sae_token_features": str(stage_dir / "sae_token_features.npz"),
                "token_to_sequence_map": str(stage_dir / "token_to_sequence_map.npz"),
                "preprocess_config": str(preprocess_config_path),
                "infer_config": str(infer_config_path),
            },
        },
    )
    return {"status": "completed", "summary_path": str(summary_path)}
