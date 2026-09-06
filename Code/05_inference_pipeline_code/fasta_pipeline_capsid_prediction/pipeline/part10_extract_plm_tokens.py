from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from .common import (
    ChunkContext,
    PipelineError,
    canonical_python_executable,
    ensure_dir,
    is_complete,
    read_csv_rows,
    run_command,
    save_part_summary,
    write_csv,
)


def _require_backend_outputs(output_dir: Path) -> None:
    required = [
        output_dir / "sample_metadata_filtered.csv",
        output_dir / "seq_token_offsets.npy",
        output_dir / "seq_token_len.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    token_layers = output_dir / "token_layers"
    if not token_layers.exists() or not any(token_layers.glob("*.tokens.npy")):
        missing.append(str(token_layers / "*.tokens.npy"))
    if missing:
        raise PipelineError(f"Embedding backend outputs missing: {missing}")


def run(
    ctx: ChunkContext, config: dict[str, Any], force: bool = False
) -> dict[str, Any]:
    stage_dir = ensure_dir(ctx.stage_dir("embed"))
    summary_path = stage_dir / "embed.summary.json"
    required_outputs = [
        stage_dir / "sequence_manifest.filtered.csv",
        stage_dir / "seq_token_offsets.npy",
        stage_dir / "seq_token_len.npy",
    ]
    if not force and is_complete(summary_path, required_outputs):
        token_layer_dir = stage_dir / "token_layers"
        if token_layer_dir.exists() and any(token_layer_dir.glob("*.tokens.npy")):
            return {"status": "skipped", "summary_path": str(summary_path)}

    input_manifest = ctx.stage_dir("prepare") / "sequence_manifest.csv"
    if not input_manifest.exists():
        raise PipelineError(f"Prepare manifest missing: {input_manifest}")

    backend_cfg = config["backends"]["embed"]
    backend_output = ensure_dir(stage_dir / "backend_output")
    backend_metadata = stage_dir / "backend_input_metadata.csv"
    rows = read_csv_rows(input_manifest)
    backend_rows = [
        {
            "prot_id": row["Seq_ID"],
            "seq": row["seq"],
            "label": config["pipeline"].get("dummy_label", "__inference__"),
        }
        for row in rows
    ]
    write_csv(backend_metadata, backend_rows, ["prot_id", "seq", "label"])

    command = [
        canonical_python_executable(config, backend_cfg),
        str(Path(backend_cfg["script_path"])),
        "--metadata",
        str(backend_metadata),
        "--output",
        str(backend_output),
        "--model",
        str(backend_cfg["model"]),
        "--layers",
        str(backend_cfg.get("layers", "28")),
        "--max-len",
        str(backend_cfg.get("max_len", 786)),
        "--bs",
        str(backend_cfg.get("batch_size", 32)),
        "--min-samples-per-class",
        "1",
    ]
    if backend_cfg.get("hf_cache"):
        command.extend(["--hf-cache", str(backend_cfg["hf_cache"])])
    if backend_cfg.get("offline"):
        command.append("--offline")
    if backend_cfg.get("trust_remote_code"):
        command.append("--trust-remote-code")
    if backend_cfg.get("bucket_sort"):
        command.append("--bucket-sort")
    if backend_cfg.get("device"):
        command.extend(["--device", str(backend_cfg["device"])])
    if backend_cfg.get("sample_frac") is not None:
        command.extend(["--sample-frac", str(backend_cfg["sample_frac"])])

    run_command(command, stage_dir, summary_path.name)
    _require_backend_outputs(backend_output)

    shutil.copy2(
        backend_output / "sample_metadata_filtered.csv",
        stage_dir / "sequence_manifest.filtered.csv",
    )
    shutil.copy2(
        backend_output / "seq_token_offsets.npy", stage_dir / "seq_token_offsets.npy"
    )
    shutil.copy2(backend_output / "seq_token_len.npy", stage_dir / "seq_token_len.npy")

    token_layers_dst = ensure_dir(stage_dir / "token_layers")
    for item in token_layers_dst.glob("*"):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    for src in (backend_output / "token_layers").glob("*.tokens.npy"):
        shutil.copy2(src, token_layers_dst / src.name)

    filtered_rows = 0
    with (stage_dir / "sequence_manifest.filtered.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        filtered_rows = sum(1 for _ in csv.DictReader(handle))

    save_part_summary(
        summary_path,
        "embed",
        {
            "chunk_id": ctx.chunk_id,
            "backend_output": str(backend_output),
            "filtered_sequence_count": filtered_rows,
            "outputs": {
                "sequence_manifest_filtered": str(
                    stage_dir / "sequence_manifest.filtered.csv"
                ),
                "seq_token_offsets": str(stage_dir / "seq_token_offsets.npy"),
                "seq_token_len": str(stage_dir / "seq_token_len.npy"),
                "token_layers_dir": str(token_layers_dst),
            },
        },
    )
    return {"status": "completed", "summary_path": str(summary_path)}
