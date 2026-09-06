#!/usr/bin/env python3
"""Run the archived implementation for this public entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "predict_property_heads_layer33.py"
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")
