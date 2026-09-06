#!/bin/bash
#SBATCH --job-name=sae_pre150k
#SBATCH --output=/path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/logs/sae_pre150k_%j.out
#SBATCH --error=/path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/logs/sae_pre150k_%j.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00

set -euo pipefail
module purge
module load miniconda3/24.1.2
source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate /path/to/hpc_home/miniconda3/envs/esm

python /path/to/vicapsid_data/SAE/preprocess_sae_training_samples.py   --input_dir /path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/subset_backend   --meta_path /path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/subset_backend/sample_metadata_filtered.csv   --layer_idx 28   --output_dir /path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/preprocessed   --epochs 15   --samples_per_epoch 4000000   --batch_size 30240   --gen_init_data   --init_samples 200000   --dtype float16   --use_staging   --staging_dir /tmp   --parallel_epochs 2   --num_threads 12
