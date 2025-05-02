#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --output=logs/gemma2b_first_last.out
#SBATCH --error=logs/gemma2b_first_last.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:2     # Request 4 GPUs (you can adjust based on availability)
#SBATCH --cpus-per-task=4   # Use 16 CPU cores
#SBATCH --mem=100000         # Request sufficient memory (adjust as needed)

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run your experiment
python run_exp.py --model_name "google/gemma-2-2b" --duplications "[(0,1),(25,1)]"

echo "Job done!"