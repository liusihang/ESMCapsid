from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_EXPERIMENT_ROOT = (
    "/path/to/vicapsid_data/PaperPart1/EmbeddingLinearProbe_cellular150k"
)
DEFAULT_MODEL_NAME = "ESMCapsid-S"
CONDA_ENV = "/path/to/hpc_home/miniconda3/envs/esm"


def dual_stage_script_name(manifest: dict[str, object]) -> str:
    suffix = f"_{manifest['result_tag']}" if manifest.get("use_fixed_layers") else ""
    return f"dual_stage_{manifest['slug']}{suffix}.sbatch"


def export_script_name(manifest: dict[str, object]) -> str:
    suffix = f"_{manifest['result_tag']}" if manifest.get("use_fixed_layers") else ""
    return f"export_{manifest['slug']}{suffix}.sbatch"


def submit_script_name(manifest: dict[str, object]) -> str:
    suffix = f"_{manifest['result_tag']}" if manifest.get("use_fixed_layers") else ""
    return f"submit_{manifest['slug']}{suffix}.sh"


def build_manifest(
    experiment_root: str = DEFAULT_EXPERIMENT_ROOT,
    model_name: str = DEFAULT_MODEL_NAME,
    result_tag: str = "v2",
    hidden_dims: list[int] | None = None,
    fixed_early_layer: int | None = None,
    fixed_late_layer: int | None = None,
    fixed_stage1_min_recall: float = 0.99,
    fixed_stage1_selection_mode: str = "min_fpr",
    fixed_stage2_max_fpr: float = 0.000025,
    fixed_stage2_selection_mode: str = "max_recall",
    parallel_folds: int = 1,
    batch_size: int = 4096,
) -> dict[str, object]:
    root = experiment_root.rstrip("/")
    slug = model_name.replace("/", "_")
    embedding_dir = f"{root}/embeddings/{slug}"
    scripts_dir = f"{root}/scripts"
    logs_dir = f"{root}/logs"
    scan_results_dir = f"{root}/results_layer_scan_{result_tag}/{slug}"
    dual_stage_results_dir = f"{root}/results_dual_stage_{result_tag}/{slug}"
    paper_tables_dir = f"{root}/paper_tables"
    use_fixed_layers = fixed_early_layer is not None and fixed_late_layer is not None
    selected_layers_json = (
        f"{scripts_dir}/selected_layers_{slug}_{result_tag}.json"
        if use_fixed_layers
        else f"{scan_results_dir}/selected_layers.json"
    )
    return {
        "experiment_root": root,
        "model_name": model_name,
        "slug": slug,
        "conda_env": CONDA_ENV,
        "scripts_dir": scripts_dir,
        "logs_dir": logs_dir,
        "embedding_dir": embedding_dir,
        "metadata": f"{embedding_dir}/sample_metadata_filtered.csv",
        "scan_results_dir": scan_results_dir,
        "dual_stage_results_dir": dual_stage_results_dir,
        "paper_tables_dir": paper_tables_dir,
        "selected_layers_json": selected_layers_json,
        "layers": list(range(4, 35)),
        "result_tag": result_tag,
        "hidden_dims": [int(dim) for dim in (hidden_dims or [512])],
        "use_fixed_layers": use_fixed_layers,
        "fixed_early_layer": fixed_early_layer,
        "fixed_late_layer": fixed_late_layer,
        "stage1_min_recall": fixed_stage1_min_recall,
        "stage1_selection_mode": fixed_stage1_selection_mode,
        "stage2_max_fpr": fixed_stage2_max_fpr,
        "stage2_selection_mode": fixed_stage2_selection_mode,
        "parallel_folds": int(parallel_folds),
        "batch_size": int(batch_size),
    }


def split_layers_evenly(layers: list[int], chunk_count: int) -> list[list[int]]:
    if chunk_count < 1:
        raise ValueError("chunk_count must be >= 1")

    base, remainder = divmod(len(layers), chunk_count)
    chunks = []
    cursor = 0
    for part_index in range(chunk_count):
        take = base + (1 if part_index < remainder else 0)
        chunks.append(layers[cursor : cursor + take])
        cursor += take
    return [chunk for chunk in chunks if chunk]


def render_layer_scan_sbatch(
    manifest: dict[str, object], part_index: int, layers: list[int]
) -> str:
    layer_arg = ",".join(str(layer) for layer in layers)
    return f"""#!/bin/bash
#SBATCH --job-name=dlscan_{manifest["slug"]}_p{part_index}
#SBATCH --output={manifest["logs_dir"]}/{manifest["slug"]}_scan_p{part_index}_%j.out
#SBATCH --error={manifest["logs_dir"]}/{manifest["slug"]}_scan_p{part_index}_%j.err
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=12:00:00

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

mkdir -p "{manifest["logs_dir"]}" "{manifest["scan_results_dir"]}"

python "{manifest["scripts_dir"]}/run_binary_layer_scan.py" \\
  --embedding-dir "{manifest["embedding_dir"]}" \\
  --metadata "{manifest["metadata"]}" \\
  --output-root "{manifest["scan_results_dir"]}" \\
  --model-name "{manifest["model_name"]}" \\
  --layers "{layer_arg}" \\
  --folds 5 \\
  --epochs 30 \\
  --lr 0.005 \\
  --weight-decay 0.0001 \\
  --batch-size 2048 \\
  --threshold-grid-size 2001 \\
  --seed 42 \\
  --device cuda
"""


def render_selection_sbatch(manifest: dict[str, object]) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=dlsel_{manifest["slug"]}
#SBATCH --output={manifest["logs_dir"]}/{manifest["slug"]}_select_%j.out
#SBATCH --error={manifest["logs_dir"]}/{manifest["slug"]}_select_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

module purge
module load miniconda3/24.1.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {manifest["conda_env"]}

export PYTHONUNBUFFERED=1

mkdir -p "{manifest["scan_results_dir"]}" "{manifest["paper_tables_dir"]}"

python "{manifest["scripts_dir"]}/summarize_binary_layer_scan.py" \\
  --scan-root "{manifest["scan_results_dir"]}" \\
  --model-name "{manifest["model_name"]}" \\
  --selected-layers-json "{manifest["selected_layers_json"]}" \\
  --early-fraction 0.5 \\
  --late-fraction 0.5 \\
  --early-metric "recall_at_fpr_0_001" \\
  --late-metric "precision_at_fpr_0_0001"
"""


def render_dual_stage_sbatch(manifest: dict[str, object]) -> str:
    hidden_dims_arg = " ".join(str(dim) for dim in manifest.get("hidden_dims", [512]))
    return f"""#!/bin/bash
#SBATCH --job-name=dlclf_{manifest["slug"]}
#SBATCH --output={manifest["logs_dir"]}/{manifest["slug"]}_dual_%j.out
#SBATCH --error={manifest["logs_dir"]}/{manifest["slug"]}_dual_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=16:00:00

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

mkdir -p "{manifest["logs_dir"]}" "{manifest["dual_stage_results_dir"]}"

if [ -f "{manifest["selected_layers_json"]}" ]; then
  cp "{manifest["selected_layers_json"]}" "{manifest["dual_stage_results_dir"]}/selected_layers.json"
fi

python "{manifest["scripts_dir"]}/train_dual_stage_low_fpr.py" \\
  --embedding-dir "{manifest["embedding_dir"]}" \\
  --metadata "{manifest["metadata"]}" \\
  --selection-json "{manifest["selected_layers_json"]}" \\
  --output-root "{manifest["dual_stage_results_dir"]}" \\
  --model-name "{manifest["model_name"]}" \\
  --folds 5 \\
  --epochs 40 \\
  --lr 0.001 \\
  --weight-decay 0.0001 \\
  --batch-size {manifest["batch_size"]} \\
  --hidden-dims {hidden_dims_arg} \\
  --calibration-fraction 0.2 \\
  --stage1-min-recall {manifest["stage1_min_recall"]} \\
  --stage1-selection-mode {manifest["stage1_selection_mode"]} \\
  --stage2-max-fpr {manifest["stage2_max_fpr"]} \\
  --stage2-selection-mode {manifest["stage2_selection_mode"]} \\
  --stage2-mode ultralowfpr \\
  --stage2-negative-multiplier 8 \\
  --min-hard-negatives 20000 \\
  --threshold-grid-size 10001 \\
  --seed 42 \\
  --device cuda \\
  --parallel-folds {manifest["parallel_folds"]}
"""


def render_export_sbatch(manifest: dict[str, object]) -> str:
    export_root = f"{manifest['dual_stage_results_dir']}/final_export"
    hidden_dims_arg = " ".join(str(dim) for dim in manifest.get("hidden_dims", [512]))
    return f"""#!/bin/bash
#SBATCH --job-name=dlexp_{manifest["slug"]}
#SBATCH --output={manifest["logs_dir"]}/{manifest["slug"]}_export_%j.out
#SBATCH --error={manifest["logs_dir"]}/{manifest["slug"]}_export_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=12:00:00

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

mkdir -p "{manifest["logs_dir"]}" "{export_root}"

python "{manifest["scripts_dir"]}/export_dual_stage_classifier.py" \\
  --embedding-dir "{manifest["embedding_dir"]}" \\
  --metadata "{manifest["metadata"]}" \\
  --selection-json "{manifest["selected_layers_json"]}" \\
  --output-root "{export_root}" \\
  --model-name "{manifest["model_name"]}" \\
  --epochs 40 \\
  --lr 0.001 \\
  --weight-decay 0.0001 \\
  --batch-size {manifest["batch_size"]} \\
  --hidden-dims {hidden_dims_arg} \\
  --stage1-min-recall {manifest["stage1_min_recall"]} \\
  --stage1-selection-mode {manifest["stage1_selection_mode"]} \\
  --stage2-max-fpr {manifest["stage2_max_fpr"]} \\
  --stage2-selection-mode {manifest["stage2_selection_mode"]} \\
  --stage2-mode ultralowfpr \\
  --stage2-negative-multiplier 8 \\
  --min-hard-negatives 20000 \\
  --threshold-grid-size 10001 \\
  --seed 42 \\
  --device cuda
"""


def render_submit_helper(manifest: dict[str, object], chunk_count: int) -> str:
    if manifest.get("use_fixed_layers"):
        return f"""#!/bin/bash
set -euo pipefail

cd "{manifest["scripts_dir"]}"

mkdir -p "{manifest["dual_stage_results_dir"]}"
dual_job=$(sbatch {dual_stage_script_name(manifest)} | awk '{{print $4}}')
echo "DUAL_STAGE $dual_job"
export_job=$(sbatch --dependency=afterok:${{dual_job}} {export_script_name(manifest)} | awk '{{print $4}}')
echo "EXPORT $export_job"

cat > "{manifest["dual_stage_results_dir"]}/submission.tsv" <<EOF
stage\tjob_id
dual_stage\t${{dual_job}}
export\t${{export_job}}
EOF
"""
    return f"""#!/bin/bash
set -euo pipefail

cd "{manifest["scripts_dir"]}"

declare -a scan_jobs=()
scan_tsv="{manifest["scan_results_dir"]}/scan_submission.tsv"
mkdir -p "{manifest["scan_results_dir"]}" "{manifest["dual_stage_results_dir"]}"
printf "part\\tjob_id\\n" > "$scan_tsv"
for script in scan_{manifest["slug"]}_part*.sbatch; do
  job_id=$(sbatch "$script" | awk '{{print $4}}')
  echo "SCAN $script $job_id"
  scan_jobs+=("$job_id")
  printf "%s\\t%s\\n" "$script" "$job_id" >> "$scan_tsv"
done

scan_dependency=$(IFS=:; echo "${{scan_jobs[*]}}")
select_job=$(sbatch --dependency=afterok:${{scan_dependency}} summarize_{manifest["slug"]}.sbatch | awk '{{print $4}}')
echo "SELECT $select_job"
dual_job=$(sbatch --dependency=afterok:${{select_job}} {dual_stage_script_name(manifest)} | awk '{{print $4}}')
echo "DUAL_STAGE $dual_job"
export_job=$(sbatch --dependency=afterok:${{dual_job}} {export_script_name(manifest)} | awk '{{print $4}}')
echo "EXPORT $export_job"

cat > "{manifest["dual_stage_results_dir"]}/submission.tsv" <<EOF
stage\tjob_id
select\t${{select_job}}
dual_stage\t${{dual_job}}
export\t${{export_job}}
EOF
"""


def write_local_bundle(
    output_dir: str, manifest: dict[str, object], chunk_count: int = 4
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scripts_dir = output_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(layer) for layer in manifest["layers"]]
    chunks = split_layers_evenly(layers, chunk_count=chunk_count)

    (output_path / "manifest.json").write_text(
        json.dumps({**manifest, "layer_chunks": chunks}, indent=2) + "\n",
        encoding="utf-8",
    )

    if manifest.get("use_fixed_layers"):
        selected_layers = {
            "early_layer": int(manifest["fixed_early_layer"]),
            "late_layer": int(manifest["fixed_late_layer"]),
            "selection_method": "fixed_layers",
            "result_tag": manifest["result_tag"],
        }
        (output_path / "selected_layers.json").write_text(
            json.dumps(selected_layers, indent=2) + "\n",
            encoding="utf-8",
        )
        (scripts_dir / Path(str(manifest["selected_layers_json"])).name).write_text(
            json.dumps(selected_layers, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        for part_index, chunk in enumerate(chunks, start=1):
            (
                scripts_dir / f"scan_{manifest['slug']}_part{part_index}.sbatch"
            ).write_text(
                render_layer_scan_sbatch(manifest, part_index=part_index, layers=chunk),
                encoding="utf-8",
            )

        (scripts_dir / f"summarize_{manifest['slug']}.sbatch").write_text(
            render_selection_sbatch(manifest),
            encoding="utf-8",
        )
    (scripts_dir / dual_stage_script_name(manifest)).write_text(
        render_dual_stage_sbatch(manifest),
        encoding="utf-8",
    )
    (scripts_dir / export_script_name(manifest)).write_text(
        render_export_sbatch(manifest),
        encoding="utf-8",
    )
    (scripts_dir / submit_script_name(manifest)).write_text(
        render_submit_helper(manifest, chunk_count=chunk_count),
        encoding="utf-8",
    )

    helper_scripts = [
        Path("/path/to/hpc_tools/experiment_tools/prepare_dual_layer_low_fpr_jobs.py"),
        Path("/path/to/hpc_tools/experiment_tools/run_binary_layer_scan.py"),
        Path("/path/to/hpc_tools/experiment_tools/summarize_binary_layer_scan.py"),
        Path("/path/to/hpc_tools/experiment_tools/train_dual_stage_low_fpr.py"),
        Path("/path/to/hpc_tools/experiment_tools/export_dual_stage_classifier.py"),
        Path("/path/to/hpc_tools/experiment_tools/predict_dual_stage_classifier.py"),
        Path("/path/to/hpc_tools/experiment_tools/layer_selection_utils.py"),
        Path("/path/to/hpc_tools/experiment_tools/linear_probe_cv_gpu.py"),
        Path("/path/to/hpc_tools/experiment_tools/linear_probe_utils.py"),
    ]
    for script in helper_scripts:
        if script.exists():
            (scripts_dir / script.name).write_text(
                script.read_text(encoding="utf-8"), encoding="utf-8"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--result-tag", default="v2")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--fixed-early-layer", type=int, default=None)
    parser.add_argument("--fixed-late-layer", type=int, default=None)
    parser.add_argument("--stage1-min-recall", type=float, default=0.99)
    parser.add_argument(
        "--stage1-selection-mode",
        choices=["min_fpr", "max_precision"],
        default="min_fpr",
    )
    parser.add_argument("--stage2-max-fpr", type=float, default=0.000025)
    parser.add_argument(
        "--stage2-selection-mode",
        choices=["max_recall", "max_precision"],
        default="max_recall",
    )
    parser.add_argument("--parallel-folds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--local-output", required=True)
    parser.add_argument("--chunk-count", type=int, default=4)
    args = parser.parse_args()

    manifest = build_manifest(
        experiment_root=args.experiment_root,
        model_name=args.model_name,
        result_tag=args.result_tag,
        hidden_dims=args.hidden_dims,
        fixed_early_layer=args.fixed_early_layer,
        fixed_late_layer=args.fixed_late_layer,
        fixed_stage1_min_recall=args.stage1_min_recall,
        fixed_stage1_selection_mode=args.stage1_selection_mode,
        fixed_stage2_max_fpr=args.stage2_max_fpr,
        fixed_stage2_selection_mode=args.stage2_selection_mode,
        parallel_folds=args.parallel_folds,
        batch_size=args.batch_size,
    )
    write_local_bundle(args.local_output, manifest, chunk_count=args.chunk_count)


if __name__ == "__main__":
    main()
