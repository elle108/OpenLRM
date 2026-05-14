#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=gan_metrics
#SBATCH --output=logs/gan_metrics_%j.out
#SBATCH --error=logs/gan_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu

echo "Job started on $(hostname) at $(date)"

source ~/.bashrc
conda activate antler_env
cd /mnt/home/milleel7/antler_project/

# Testing 4x scale
echo "Running 4x scale experiment"
python eval_metrics.py \
    --input /mnt/home/milleel7/antler_project/Antler-Super-resolution.jpg \
    --model /mnt/home/milleel7/antler_project/RealESRGAN_x4plus.pth \
    --scale 4

# Testing 8x scale
echo "Running 8x scale experiment"
python eval_metrics.py \
    --input /mnt/home/milleel7/antler_project/Antler-Super-resolution.jpg \
    --model /mnt/home/milleel7/antler_project/RealESRGAN_x4plus.pth \
    --scale 8

echo "Job completed at $(date)"