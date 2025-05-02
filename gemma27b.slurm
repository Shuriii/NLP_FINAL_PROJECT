#!/bin/bash
#SBATCH --job-name=gemma27b
#SBATCH --output=logs/gemma27b_vllm.out
#SBATCH --error=logs/gemma27b_vllm.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint="geforce_rtx_3090"
#SBATCH --gres=gpu:4     # Request 4 GPUs (you can adjust based on availability)
#SBATCH --cpus-per-task=4   # Use 16 CPU cores
#SBATCH --mem=257000          # Request sufficient memory (adjust as needed)

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

# Set Hugging Face cache directory
export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run your experiment
python run_exp_large.py --model_name "google/gemma-2-27b"

echo "Job done!"