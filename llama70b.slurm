#! /bin/bash

#SBATCH --job-name=llama70b
#SBATCH --output=logs/llama70b.out
#SBATCH --error=logs/llama70b.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=50000
#SBATCH --nodes=1
#SBATCH --gpus=4

echo "Starting job"

# Load conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

# Set Hugging Face cache to the regular directory (for models)
export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers

# Run the generation script
python run_all_models.py "meta-llama/Llama-3-70b"

echo "Done!"

