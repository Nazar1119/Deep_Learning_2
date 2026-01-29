#!/bin/bash
#SBATCH --job-name=internvl3_5_eval
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source venvs/vlm-env/bin/activate

echo "Running on $(hostname)"
nvidia-smi

python modules/scripts/intervl3.5-30B-A3B.py
