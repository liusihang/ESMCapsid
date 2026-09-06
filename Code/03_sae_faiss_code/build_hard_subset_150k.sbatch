#!/bin/bash
#SBATCH --job-name=sae_sub150k
#SBATCH --output=/path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/logs/sae_sub150k_%j.out
#SBATCH --error=/path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/logs/sae_sub150k_%j.err
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

python /path/to/vicapsid_data/SAE/retrain_d4604k64_hardsub150k_20260404_164446/scripts/build_hard_subset_150k.py
