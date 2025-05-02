import json
import os

# ===> Input path to your reformatted DROP samples
input_path = "data_sets/drop/drop_input_prompt_ samples/drop_reformatted_samples.json"
output_path = "eval/golden_ans/drop_gold_ans.json"

# Load the reformatted DROP samples
with open(input_path, "r") as f:
    samples = json.load(f)

# Extract answers from output.answers_spans.spans
gold_answers = {}
for sample in samples:
    rid = str(sample["research_id"])
    spans = sample["output"]["answers_spans"]["spans"]
    gold_answers[rid] = spans  # list of acceptable answers

# Ensure output folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to JSON
with open(output_path, "w") as f:
    json.dump(gold_answers, f, indent=2)

print(f"Golden answers saved to: {output_path}")
