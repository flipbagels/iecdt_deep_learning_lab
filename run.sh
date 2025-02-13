#!/bin/bash
#SBATCH --job-name=DiffusionTest
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

conda activate semtwo
python3 train_vit.py