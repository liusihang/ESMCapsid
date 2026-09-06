from __future__ import annotations

from pathlib import Path


SAE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = SAE_ROOT / "results"
CLUSTER_CSV = SAE_ROOT / "SAE_mean_cluster.csv"
MOTIF_PARQUET = SAE_ROOT / "viral_capsid_reindexed.parquet"
MAPPING_CSV = SAE_ROOT / "motif_id_reindex_mapping.csv"
MOTIF_CLUSTER_CONSERVATION_DIR = RESULTS_DIR / "motif_cluster_conservation"
ASSIGNMENT_CSV = (
    MOTIF_CLUSTER_CONSERVATION_DIR / "metadata" / "Finial_Cluster_assigment.csv"
)


def ensure_sae_on_syspath() -> None:
    import sys

    sae_root_str = str(SAE_ROOT)
    if sae_root_str not in sys.path:
        sys.path.insert(0, sae_root_str)
