from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def parser():
    result = argparse.ArgumentParser(description="ESM-only inference for public ESMCapsid models")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="mode", required=True)
    for name, description in [
        ("predict", "Screen proteins with S, then encode candidates with C"),
        ("embed", "Encode supplied proteins with public C (no screening or property claims)"),
        ("annotate", "Run the released property heads on supplied candidate proteins"),
    ]:
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--input", type=Path, required=True, help="Protein FASTA/FASTA.gz or CSV/CSV.gz")
        command.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are refused")
        command.add_argument("--models", type=Path, help="Local parent containing ESMCapsid-S and/or ESMCapsid-C")
        command.add_argument("--offline", action="store_true", help="Use local directories or existing HF cache only")
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--sequence-column", default="seq", help="CSV sequence column")
        command.add_argument("--id-column", default="prot_id", help="CSV ID column")
        command.add_argument("--c-max-tokens", type=int, default=786, help="C tokenizer limit, including special tokens")
        if name == "predict":
            command.add_argument("--screen-only", action="store_true")
            command.add_argument("--save-embeddings", action="store_true", help="Also save S embeddings; C vectors are always saved")
            command.add_argument("--screen-max-tokens", type=int, default=1022)
        if name != "embed":
            command.add_argument("--property-heads", type=Path, help="Local property-head directory (released on HF under ESMCapsid-C/heads/property_heads)")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .pipeline import RunOptions, run

        summary = run(RunOptions(**vars(args)))
    except Exception as exc:
        print(f"ESMCapsid error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": summary["status"], "output": str(args.out.resolve()),
        "input_count": summary["input_count"], "invalid_count": summary["invalid_count"],
        "candidate_count": summary.get("candidate_count"),
        "embedding_count": summary.get("embedding_count"), "warnings": summary["warnings"],
    }, indent=2, ensure_ascii=False))
    return 0
