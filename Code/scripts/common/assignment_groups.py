from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.common.sae_paths import ASSIGNMENT_CSV, MOTIF_CLUSTER_CONSERVATION_DIR


OUTPUT_ROOT = MOTIF_CLUSTER_CONSERVATION_DIR
METADATA_DIR = OUTPUT_ROOT / "metadata"
TARGET_GROUPS = ("HK97_like", "Pico_like", "NCLDV-like")


def load_analysis_groups(
    assignment_csv: Path = ASSIGNMENT_CSV,
    target_groups: tuple[str, ...] = TARGET_GROUPS,
) -> dict[str, list[int]]:
    df = pd.read_csv(assignment_csv)
    required_columns = {"group", "cluster"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Assignment CSV missing columns: {sorted(missing)}")

    df = df.dropna(subset=["group", "cluster"]).copy()
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    groups: dict[str, list[int]] = {}
    for group_name in target_groups:
        cluster_ids = sorted(
            df.loc[df["group"] == group_name, "cluster"].unique().tolist()
        )
        if not cluster_ids:
            raise ValueError(
                f"No clusters found for group {group_name} in {assignment_csv}"
            )
        groups[group_name] = cluster_ids
    return groups
