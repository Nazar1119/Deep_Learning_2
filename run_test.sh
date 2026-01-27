#!/bin/bash
#SBATCH --job-name=deepseek_text
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export HF_HOME=/mnt/data/natyke582/hf

source venvs/vlm-env/bin/activate

echo "Running on $(hostname)"
nvidia-smi

python modules/scripts/deepseek-70b.py
