#!/bin/bash
#SBATCH --job-name=llama70b
#SBATCH --output=logs/llama70b.out
#SBATCH --error=logs/llama70b.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint="geforce-rtx-3090"   # Request 4 GPUs (you can adjust based on availability)
#SBATCH --gres=gpu:4     # Request 4 GPUs (you can adjust based on availability)
#SBATCH --cpus-per-task=4    # Use 16 CPU cores
#SBATCH --mem=257000           # Request sufficient memory (adjust as needed)

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

# Set Hugging Face cache directory
export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers

# Run your experiment
python run_exp_large.py --model_name "meta-llama/Meta-Llama-3-70B"

echo "Job done!"


