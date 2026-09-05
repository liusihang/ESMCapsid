from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_EXPERIMENT_ROOT = "/path/to/vicapsid_data/PaperPart1/EmbeddingLinearProbe"
CONDA_ENV = "/path/to/hpc_home/miniconda3/envs/esm"


def default_cluster_map(experiment_root: str = DEFAULT_EXPERIMENT_ROOT) -> str:
    return f"{experiment_root}/low_identity_clusters/capsid_cluster_map.csv"


def build_model_specs(experiment_root: str = DEFAULT_EXPERIMENT_ROOT):
    return {
        "ESMplusplus_large": {
            "embedding": f"{experiment_root}/embeddings/ESMplusplus_large/layer_34.npy",
            "metadata": f"{experiment_root}/embeddings/ESMplusplus_large/sample_metadata_filtered.csv",
        },
        "FastESM2_650M": {
            "embedding": f"{experiment_root}/embeddings/FastESM2_650M/layer_31.npy",
            "metadata": f"{experiment_root}/embeddings/FastESM2_650M/sample_metadata_filtered.csv",
        },
        "Profluent-E1-600M": {
            "embedding": f"{experiment_root}/embeddings/Profluent-E1-600M/layer_28.npy",
            "metadata": f"{experiment_root}/embeddings/Profluent-E1-600M/sample_metadata_filtered.csv",
        },
        "esm3o": {
            "embedding": f"{experiment_root}/embeddings/esm3o/layer_46.npy",
            "metadata": f"{experiment_root}/embeddings/esm3o/sample_metadata_filtered.csv",
        },
        "merged_full_model": {
            "embedding": f"{experiment_root}/embeddings/merged_full_model/layer_34.npy",
            "metadata": f"{experiment_root}/embeddings/merged_full_model/sample_metadata_filtered.csv",
        },
        "ESMCapsid-S": {
            "embedding": f"{experiment_root}/embeddings/ESMCapsid-S/layer_34.npy",
            "metadata": f"{experiment_root}/embeddings/ESMCapsid-S/sample_metadata_filtered.csv",
        },
    }


def render_sbatch(
    model_name: str,
    embedding_path: str,
    metadata_path: str,
    experiment_root: str = DEFAULT_EXPERIMENT_ROOT,
    cluster_map: str | None = None,
) -> str:
    slug = model_name
    output_dir = f"{experiment_root}/results_low_identity_30/{slug}"
    logs_dir = f"{experiment_root}/logs"
    script_path = f"{experiment_root}/scripts/run_low_identity_capsid.py"
    cluster_map = cluster_map or default_cluster_map(experiment_root)
    return f"""#!/bin/bash
#SBATCH --job-name=lowid30_{slug}
#SBATCH --output={logs_dir}/{slug}_lowid30_%j.out
#SBATCH --error={logs_dir}/{slug}_lowid30_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=12:00:00

set -euo pipefail

module purge
module load miniconda3/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {CONDA_ENV}

export PYTHONUNBUFFERED=1

mkdir -p "{output_dir}"

python "{script_path}" \\
  --embedding "{embedding_path}" \\
  --metadata "{metadata_path}" \\
  --cluster-map "{cluster_map}" \\
  --output "{output_dir}" \\
  --model-name "{model_name}" \\
  --identity-threshold 0.3 \\
  --folds 5 \\
  --negative-test-frac 0.2 \\
  --epochs 30 \\
  --lr 0.005 \\
  --weight-decay 0.0001 \\
  --batch-size 2048 \\
  --seed 42 \\
  --device cuda
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-output", required=True)
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--cluster-map", default=None)
    args = parser.parse_args()

    output_dir = Path(args.local_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, spec in build_model_specs(args.experiment_root).items():
        (output_dir / f"lowid_30_{model_name}.sbatch").write_text(
            render_sbatch(
                model_name,
                spec["embedding"],
                spec["metadata"],
                experiment_root=args.experiment_root,
                cluster_map=args.cluster_map,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
