#!/bin/bash
#SBATCH --job-name=llama_s
#SBATCH --output=logs/llama_s.out
#SBATCH --error=logs/llama_s.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint="geforce_rtx_3090"   # Request 4 GPUs (you can adjust based on availability)
#SBATCH --gpus=2     # Request 2 GPUs (you can adjust based on availability)
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your experiment
python run_exp.py --model_name "meta-llama/Meta-Llama-3-8B" --duplications '[(27,1),(28,1),(29,1),(30,1),(31,1)]'

echo "Job done!"
