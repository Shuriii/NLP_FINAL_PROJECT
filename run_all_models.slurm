#!/bin/bash
#SBATCH --job-name=run_all_models
#SBATCH --output=logs/run_all_models_%j.out
#SBATCH --error=logs/run_all_models_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

echo "Starting job"

# Load conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

# Set Hugging Face cache to the regular directory (for models)
export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers

MODEL_NAME=$1

python run_all_models.py $MODEL_NAME

echo "Done!"
