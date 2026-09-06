#!/bin/bash

#SBATCH --job-name=vicapsid_sae
#SBATCH --output=sae_ft_%j.out
#SBATCH --error=sae_ft_%j.err
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=300G
#SBATCH --gres=gpu:a800:1
#SBATCH --time=48:00:00

echo "=========================================="
echo "ViCapsid SAE training"
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Start Time:    $(date)"
echo "GPU:           $CUDA_VISIBLE_DEVICES"
echo "=========================================="

module purge
module load miniconda3/24.1.2

source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate /path/to/hpc_home/miniconda3/envs/esm

DATA_DIR="/path/to/vicapsid_data/sae/preprocessed"
LABELS_CSV="/path/to/vicapsid_data/sae/sample_metadata_filtered.csv"
OUTPUT_ROOT="/path/to/vicapsid_data/sae/output"

DIMS="4604"
KS="64"

LR=4e-4
EPOCHS=20

LOSS_TYPE="huber"
HUBER_DELTA="5"

echo "------------------------------------------"
echo "Config:"
echo "Data: $DATA_DIR"
echo "Dims: $DIMS"
echo "Ks:   $KS"
echo "Loss: $LOSS_TYPE (delta=$HUBER_DELTA)"
echo "------------------------------------------"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_sae_hybrid.py \
    --data_dir "$DATA_DIR" \
    --labels_csv "$LABELS_CSV" \
    --output_root "$OUTPUT_ROOT" \
    --dims $DIMS \
    --ks $KS \
    --epochs $EPOCHS \
    --lr $LR \
    --loss_type $LOSS_TYPE \
    --huber_delta $HUBER_DELTA \
    --stage all

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully at $(date)"
else
    echo "Job failed with exit code $EXIT_CODE at $(date)"
fi
echo "=========================================="

exit $EXIT_CODE
