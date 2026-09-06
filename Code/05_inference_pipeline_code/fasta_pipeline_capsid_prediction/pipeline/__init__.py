"""Canonical FASTA prediction pipeline package."""

PART_ORDER = [
    "prepare",
    "embed",
    "sae",
    "map_semantic_motif",
    "pool_seq",
    "predict_seq",
    "report",
]
