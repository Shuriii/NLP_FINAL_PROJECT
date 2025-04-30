#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --output=logs/gemma2b.out
#SBATCH --error=logs/gemma2b.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=geforce_rtx_3090
#SBATCH --gres=gpu:1     # Request 4 GPUs (you can adjust based on availability)
#SBATCH --cpus-per-task=4   # Use 16 CPU cores
#SBATCH --mem=257500          # Request sufficient memory (adjust as needed)

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

# Set Hugging Face cache directory
export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers

# Run your experiment
python run_exp.py --model_name "google/gemma-2-2b"

echo "Job done!"