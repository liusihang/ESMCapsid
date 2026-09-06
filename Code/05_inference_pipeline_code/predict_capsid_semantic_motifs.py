#!/usr/bin/env python3
"""Run the archived implementation for this public entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parent
        / "fasta_pipeline_capsid_prediction/run_fasta_pipeline.py"
    )
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
