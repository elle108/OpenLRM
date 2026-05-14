#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=antler_color
#SBATCH --output=logs/color_%j.out
#SBATCH --error=logs/color_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu

mkdir -p logs

echo "=========================================="
echo "Running IR Image Colorization"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load modules
ml miniconda3
ml cuda/12.8

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate antler_env

# Define paths
INPUT_IMG="2 - Colorization Example.jpg"
OUTPUT_IMG="Colorized_IR.jpg"

# Run the Python script
python colorize.py \
    --input "$INPUT_IMG" \
    --output "$OUTPUT_IMG" \
    --prototxt "colorization_deploy_v2.prototxt" \
    --model "colorization_release_v2.caffemodel" \
    --points "pts_in_hull.npy"

echo "=========================================="
echo "Finished Colorization"
echo "=========================================="