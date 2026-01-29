#!/bin/bash
#SBATCH --job-name=step3_vl_10b
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/step3_vl_10b_%j.out
#SBATCH --error=logs/step3_vl_10b_%j.err

export HF_HOME=/mnt/data/$USER/hf
export HF_HUB_CACHE=/mnt/data/$USER/hf

source venvs/vlm-env/bin/activate

python modules/scripts/step3vl-30b.py