#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=real_esrgan
#SBATCH --output=logs/realesrgan_%j.out
#SBATCH --error=logs/realesrgan_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu

mkdir -p logs

echo "=========================================="
echo "Starting Real-ESRGAN Texture Upscaling"
echo "=========================================="

ml miniconda3
ml cuda/12.8
source $(conda info --base)/etc/profile.d/conda.sh
conda activate antler_env

# --- REPLACE THESE PATHS ---
INPUT_IMG="/mnt/home/milleel7/antler_project/Antler-Super-resolution.jpg"
OUTPUT_IMG="/mnt/home/milleel7/antler_project/High_Res_GAN_Antler.jpg"
MODEL_PATH="/mnt/home/milleel7/antler_project/RealESRGAN_x4plus.pth"

python /mnt/home/milleel7/antler_project/run_realesrgan.py \
    --input "$INPUT_IMG" \
    --output "$OUTPUT_IMG" \
    --model "$MODEL_PATH"

echo "=========================================="
echo "Finished GAN Super-Resolution"
echo "=========================================="