#!/bin/bash
#SBATCH --job-name=llama_n
#SBATCH --output=logs/llama_n.out
#SBATCH --error=logs/llama_n.err
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
python run_exp_dup_chunks.py --model_name "meta-llama/Meta-Llama-3-8B" --duplications '[(14,3),(17,3)]'

echo "Job done!"
