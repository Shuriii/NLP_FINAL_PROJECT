#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --output=logs/llama8b.out
#SBATCH --error=logs/llama8b.err
#SBATCH --time=24:00:00
#SBATCH --account=gpu-research
#SBATCH --constraint=geforce_rtx_3090
#SBATCH --partition=killable
#SBATCH --gres=gpu:4     # Request 2 GPUs (you can adjust based on availability)
#SBATCH --cpus-per-task=4    # Use 16 CPU cores
#SBATCH --mem=257500           # Request sufficient memory (adjust as needed)

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

# Set Hugging Face cache directory
export HF_HOME=/home/joberant/NLP_2425a/sharonsaban/.cache/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run your experiment
python run_exp.py --model_name "meta-llama/Meta-Llama-3-8B"

echo "Job done!"