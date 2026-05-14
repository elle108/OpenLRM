#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=antler_inpaint
#SBATCH --output=logs/inpaint_%j.out
#SBATCH --error=logs/inpaint_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu

mkdir -p logs

echo "=========================================="
echo "Running Image Inpainting"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
ml miniconda3
ml cuda/12.8

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate antler_env

# Define paths
INPUT_IMG="3 - Image impending (Side View).jpg"
MASK_IMG="antler_mask.png"
OUTPUT_IMG="Inpainted_Antler.jpg"

# Run the Python script
python inpaint.py \
    --input "$INPUT_IMG" \
    --mask "$MASK_IMG" \
    --output "$OUTPUT_IMG"

echo "=========================================="
echo "Finished Inpainting"
echo "=========================================="