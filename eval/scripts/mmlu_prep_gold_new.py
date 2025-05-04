import json
import os

input_path = "data_sets/mmlu/mmlu_input_prompt_ samples/mmlu_reformatted_samples.json"
output_path = "eval/golden_ans/mmlu_gold_ans_new.json"

with open(input_path, "r") as f:
    samples = json.load(f)

gold_answers = {}
for sample in samples:
    rid = str(sample["research_id"])
    gold_answers[rid] = {
        "answer": sample["output"],
        "subject": sample["subject"],
        "super_category": sample["super_category"]
    }

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(gold_answers, f, indent=2)

print(f"Golden answers saved to: {output_path}")
