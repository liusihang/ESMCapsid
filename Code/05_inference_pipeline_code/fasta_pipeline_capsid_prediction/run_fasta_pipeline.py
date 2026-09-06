#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.common import (
    ChunkContext,
    PipelineError,
    load_json,
    normalize_part_selection,
    resolve_relative_config_paths,
    resolve_chunk_ids,
)
from pipeline.preflight import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical FASTA pipeline parts")
    parser.add_argument(
        "--fasta", type=Path, default=None, help="Required when running prepare"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--parts", required=True, help="Comma-separated part list or 'all'"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--chunk-id", default=None, help="Optional chunk id or integer chunk index"
    )
    parser.add_argument(
        "--map-backend",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Backend choice for part30_map_semantic_motif",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    resolve_relative_config_paths(config, config_path)
    selected_parts = normalize_part_selection(args.parts)
    validate_config(config, selected_parts)
    run_dir = args.run_dir.resolve()

    if "prepare" in selected_parts:
        from pipeline.part00_prepare_input import run as run_prepare

        if args.fasta is None:
            raise PipelineError("--fasta is required when running prepare")
        run_prepare(
            run_dir=run_dir,
            fasta_path=args.fasta.resolve(),
            config=config,
            force=args.force,
        )
    elif not (run_dir / "manifest" / "run_manifest.json").exists():
        raise PipelineError(
            "Run manifest missing. Run prepare first or include --parts prepare."
        )

    chunk_ids = resolve_chunk_ids(run_dir, args.chunk_id)
    for chunk_id in chunk_ids:
        ctx = ChunkContext(run_dir=run_dir, chunk_id=chunk_id)
        if "embed" in selected_parts:
            from pipeline.part10_extract_plm_tokens import run as run_embed

            run_embed(ctx=ctx, config=config, force=args.force)
        if "sae" in selected_parts:
            from pipeline.part20_infer_sae_tokens import run as run_sae

            run_sae(ctx=ctx, config=config, force=args.force)
        if "map_semantic_motif" in selected_parts:
            from pipeline.part30_map_semantic_motif import run as run_map_semantic_motif

            run_map_semantic_motif(
                ctx=ctx,
                config=config,
                map_backend=args.map_backend,
                force=args.force,
            )
        if "pool_seq" in selected_parts:
            from pipeline.part40_pool_sequence_features import run as run_pool_seq

            run_pool_seq(ctx=ctx, config=config, force=args.force)
        if "predict_seq" in selected_parts:
            from pipeline.part50_predict_sequence_knn import run as run_predict_seq

            run_predict_seq(ctx=ctx, config=config, force=args.force)

    if "report" in selected_parts:
        from pipeline.part60_build_report import run as run_report

        run_report(run_dir=run_dir, config=config, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
