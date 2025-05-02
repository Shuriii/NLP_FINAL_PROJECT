import os

template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}.out
#SBATCH --error=logs/{job_name}.err
#SBATCH --time=24:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000

echo "Starting job..."

# Activate conda environment
source /home/joberant/NLP_2425a/sharonsaban/anaconda3/etc/profile.d/conda.sh
conda activate sharon_env

export TRANSFORMERS_CACHE=/home/joberant/NLP_2425a/sharonsaban/.cache/huggingface/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your experiment
python {script_name} --model_name "{model_name}" --duplications '{layer_config}'

echo "Job done!"
"""

configs = {
    "A": {"type": "New", "Gemma": [(11, 3)], "LLaMA": [(14, 3)]},
    "B": {"type": "Old", "Gemma": [(13, 2)], "LLaMA": [(16, 2)]},
    "C": {"type": "New", "Gemma": [(10, 5)], "LLaMA": [(13, 5)]},
    "D": {"type": "New", "Gemma": [(0, 4)], "LLaMA": [(0, 5)]},
    "E": {"type": "New", "Gemma": [(22, 4)], "LLaMA": [(27, 5)]},
    "F": {"type": "New", "Gemma": [(0, 3), (22, 3)], "LLaMA": [(0, 4), (27, 4)]},
}

models = {
    "Gemma": "google/gemma-2-2b",
    "LLaMA": "meta-llama/Meta-Llama-3-8B",
}

def format_no_spaces(tuples):
    return "[" + ",".join(f"({a},{b})" for a, b in tuples) + "]"

os.makedirs("slurms", exist_ok=True)

for config_name, details in configs.items():
    script_type = "run_exp_dup_chunks.py" if details["type"] == "New" else "run_exp.py"
    
    for model_key, model_name in models.items():
        job_name = f"{model_key.lower()}_{config_name.lower()}"
        layer_config = format_no_spaces(details[model_key])

        slurm_script = template.format(
            job_name=job_name,
            script_name=script_type,
            model_name=model_name,
            layer_config=layer_config,
        )

        with open(f"{job_name}.slurm", "w") as f:
            f.write(slurm_script)

