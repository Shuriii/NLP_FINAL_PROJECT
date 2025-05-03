#!/bin/bash
#SBATCH --job-name=gemma_p
#SBATCH --output=logs/gemma_p.out
#SBATCH --error=logs/gemma_p.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your experiment
python run_exp_dup_chunks.py --model_name "google/gemma-2-2b" --duplications '[(22,2)]'

echo "Job done!"
