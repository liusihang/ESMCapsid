#!/usr/bin/env python3
"""Run the archived SAE implementation from the self-contained pipeline."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = (
        Path(__file__).resolve().parents[3]
        / "03_sae_faiss_code"
        / "train_sae_hybrid.py"
    )
    runpy.run_path(str(target), run_name="__main__")
