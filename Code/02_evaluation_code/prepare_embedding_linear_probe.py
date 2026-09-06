from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATASET_CSV = "/path/to/vicapsid_data/benchmark/sample_metadata.csv"
DEFAULT_EXPERIMENT_ROOT = "/path/to/vicapsid_data/benchmark/embedding_linear_probe"
CONDA_ENV = "/path/to/hpc_home/miniconda3/envs/esm"
ESM_EXTRACTOR = "/path/to/vicapsid_code/05_inference_pipeline_code/extract_sequence_embeddings.py"
PROFLUENT_EXTRACTOR = "/path/to/vicapsid_hpc/Scripts/extract_embeddings_profluent.py"


def build_model_specs():
    return {
        "ESMplusplus_large": {
            "extractor": "esm",
            "extractor_script": ESM_EXTRACTOR,
            "model_arg": "/path/to/project_data/db/Model/Synthyra/ESMplusplus_large",
            "penultimate_layer": 34,
            "batch_size": 16,
            "max_len": 1022,
            "embed_time": "24:00:00",
        },
        "FastESM2_650M": {
            "extractor": "esm",
            "extractor_script": ESM_EXTRACTOR,
            "model_arg": "/path/to/project_data/db/Model/Synthyra/FastESM2_650M",
            "penultimate_layer": 31,
            "batch_size": 24,
            "max_len": 1022,
            "embed_time": "24:00:00",
        },
        "Profluent-E1-600M": {
            "extractor": "profluent",
            "extractor_script": PROFLUENT_EXTRACTOR,
            "model_arg": "/path/to/project_data/db/Model/Synthyra/Profluent-E1-600M",
            "penultimate_layer": 28,
            "batch_size": 16,
            "max_len": 1022,
            "embed_time": "24:00:00",
        },
        "esm3o": {
            "extractor": "esm",
            "extractor_script": ESM_EXTRACTOR,
            "entry_script": "extract_esm3_with_registry_patch.py",
            "model_arg": "esm3_sm_open_v1",
            "penultimate_layer": 46,
            "batch_size": 1,
            "max_len": 1022,
            "embed_time": "72:00:00",
            "pre_commands": [
                "mkdir -p /path/to/project_data/db/Model/esm3o/data/weights",
                (
                    "ln -sfn "
                    "/path/to/project_data/db/Model/esm3o/esm3_sm_open_v1_full.pth "
                    "/path/to/project_data/db/Model/esm3o/data/weights/esm3_sm_open_v1.pth"
                ),
            ],
        },
        "ESMCapsid-S": {
            "extractor": "esm",
            "extractor_script": ESM_EXTRACTOR,
            "model_arg": "/path/to/models/ESMCapsid-S",
            "penultimate_layer": 34,
            "layers_arg": ",".join(str(layer) for layer in range(4, 35)),
            "batch_size": 16,
            "max_len": 1022,
            "embed_time": "24:00:00",
        },
    }


def build_remote_manifest(
    experiment_root: str = DEFAULT_EXPERIMENT_ROOT,
    dataset_csv: str = DEFAULT_DATASET_CSV,
):
    root = experiment_root.rstrip("/")
    scripts_dir = f"{root}/scripts"
    logs_dir = f"{root}/logs"
    embeddings_root = f"{root}/embeddings"
    results_root = f"{root}/results"
    models = {}

    for model_name, spec in build_model_specs().items():
        slug = model_name.replace("/", "_")
        models[model_name] = {
            **spec,
            "slug": slug,
            "embedding_dir": f"{embeddings_root}/{slug}",
            "result_dir": f"{results_root}/{slug}",
            "embedding_job": f"{scripts_dir}/embed_{slug}.sbatch",
            "probe_job": f"{scripts_dir}/probe_{slug}.sbatch",
        }

    return {
        "dataset_csv": dataset_csv,
        "experiment_root": root,
        "scripts_dir": scripts_dir,
        "logs_dir": logs_dir,
        "embeddings_root": embeddings_root,
        "results_root": results_root,
        "conda_env": CONDA_ENV,
        "models": models,
    }


def render_embedding_sbatch(manifest: dict, model_name: str) -> str:
    spec = manifest["models"][model_name]
    layers_arg = spec.get("layers_arg", str(spec["penultimate_layer"]))
    pooling_arg = spec.get("pooling_mode")
    pre_commands = "\n".join(spec.get("pre_commands", []))
    if pre_commands:
        pre_commands += "\n"
    runner_script = spec["extractor_script"]
    runner_extra = ""
    if spec.get("bundle_extractor"):
        runner_script = (
            f"{manifest['scripts_dir']}/{Path(spec['extractor_script']).name}"
        )
    if spec.get("entry_script"):
        runner_script = f"{manifest['scripts_dir']}/{spec['entry_script']}"
        runner_extra = f'  --script-path "{spec["extractor_script"]}" \\\n'
    pooling_line = f'  --pooling "{pooling_arg}" \\\n' if pooling_arg else ""

    return f"""#!/bin/bash
#SBATCH --job-name=elp_emb_{spec["slug"]}
#SBATCH --output={manifest["logs_dir"]}/{spec["slug"]}_embed_%j.out
#SBATCH --error={manifest["logs_dir"]}/{spec["slug"]}_embed_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40:1
#SBATCH --time={spec["embed_time"]}

set -euo pipefail

module purge
module load miniconda3/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {manifest["conda_env"]}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1

mkdir -p "{manifest["scripts_dir"]}" "{manifest["logs_dir"]}" "{manifest["embeddings_root"]}" "{manifest["results_root"]}" "{spec["embedding_dir"]}" "{spec["result_dir"]}"
{pre_commands}python "{runner_script}" \\
{runner_extra}  --metadata "{manifest["dataset_csv"]}" \\
  --model "{spec["model_arg"]}" \\
  --output "{spec["embedding_dir"]}" \\
  --offline \\
  --trust-remote-code \\
  --layers "{layers_arg}" \\
{pooling_line}  --sample-frac 1 \\
  --min-samples-per-class 1 \\
  --max-len {spec["max_len"]} \\
  --bs {spec["batch_size"]} \\
  --bucket-sort
"""


def render_probe_sbatch(manifest: dict, model_name: str) -> str:
    spec = manifest["models"][model_name]
    embedding_file = f"{spec['embedding_dir']}/layer_{spec['penultimate_layer']}.npy"
    probe_script = f"{manifest['scripts_dir']}/linear_probe_cv_gpu.py"
    return f"""#!/bin/bash
#SBATCH --job-name=elp_probe_{spec["slug"]}
#SBATCH --output={manifest["logs_dir"]}/{spec["slug"]}_probe_%j.out
#SBATCH --error={manifest["logs_dir"]}/{spec["slug"]}_probe_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=08:00:00

set -euo pipefail

module purge
module load miniconda3/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {manifest["conda_env"]}

export PYTHONUNBUFFERED=1

mkdir -p "{spec["result_dir"]}"

python "{probe_script}" \\
  --embedding "{embedding_file}" \\
  --metadata "{spec["embedding_dir"]}/sample_metadata_filtered.csv" \\
  --output "{spec["result_dir"]}" \\
  --model-name "{model_name}" \\
  --folds 5 \\
  --epochs 30 \\
  --lr 0.005 \\
  --weight-decay 0.0001 \\
  --batch-size 2048 \\
  --seed 42 \\
  --device cuda
"""


def write_local_bundle(output_dir: str, manifest: dict):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scripts_dir = output_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    (output_path / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for model_name in manifest["models"]:
        spec = manifest["models"][model_name]
        (scripts_dir / Path(spec["embedding_job"]).name).write_text(
            render_embedding_sbatch(manifest, model_name),
            encoding="utf-8",
        )
        (scripts_dir / Path(spec["probe_job"]).name).write_text(
            render_probe_sbatch(manifest, model_name),
            encoding="utf-8",
        )

    extra_scripts = [
        Path("/path/to/hpc_tools/experiment_tools/linear_probe_cv_gpu.py"),
        Path("/path/to/hpc_tools/experiment_tools/linear_probe_utils.py"),
        Path("/path/to/hpc_tools/experiment_tools/prepare_embedding_linear_probe.py"),
        Path("/path/to/hpc_tools/experiment_tools/extract_esm3_with_registry_patch.py"),
        Path(BUNDLED_ESM_EXTRACTOR),
    ]
    for script in extra_scripts:
        (scripts_dir / script.name).write_text(
            script.read_text(encoding="utf-8"), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--dataset-csv", default=DEFAULT_DATASET_CSV)
    parser.add_argument("--local-output", default=None)
    args = parser.parse_args()

    manifest = build_remote_manifest(
        experiment_root=args.experiment_root,
        dataset_csv=args.dataset_csv,
    )
    if args.local_output:
        write_local_bundle(args.local_output, manifest)
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
