import json
import os

# ===> Input path to your reformatted MMLU samples
input_path = "data_sets/mmlu/mmlu_input_prompt_ samples/mmlu_reformatted_samples.json"
output_path = "eval/golden_ans/mmlu_gold_ans.json"

# Load the reformatted MMLU samples
with open(input_path, "r") as f:
    samples = json.load(f)

# Build the golden answer dictionary
gold_answers = {}
for sample in samples:
    rid = str(sample["research_id"])  # convert to string to match prediction format
    gold_answers[rid] = sample["output"]

# Ensure the output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to file
with open(output_path, "w") as f:
    json.dump(gold_answers, f, indent=2)

print(f"Golden answers saved to: {output_path}")
