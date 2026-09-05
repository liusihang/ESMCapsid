#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.common import (
    load_json,
    normalize_part_selection,
    resolve_relative_config_paths,
)
from pipeline.preflight import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ViCapsid pipeline code, model, and reference paths."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--parts",
        default="all",
        help="Comma-separated pipeline parts or 'all'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    resolve_relative_config_paths(config, config_path)
    checks = validate_config(config, normalize_part_selection(args.parts))
    print(json.dumps({"ok": True, "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
