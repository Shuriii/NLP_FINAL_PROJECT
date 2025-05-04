import json
import os

# ===> Input paths
formatted_path = "data_sets/musique/musique_input_prompt_ samples/musique_reformatted_samples.json"
raw_path = "data_sets/musique/musique_raw_data_ samples/musique_500_samples_from_dev.json"  # ⬅️ You fill this in
output_path = "eval/golden_ans/musique_gold_ans.json"

# Load data
with open(formatted_path, "r") as f:
    formatted_samples = json.load(f)

with open(raw_path, "r") as f:
    raw_samples = json.load(f)

# Build lookup from raw samples using (id, answerable) as key
raw_lookup = {}
for raw in raw_samples:
    key = (raw["id"], raw["answerable"])
    raw_lookup[key] = raw

# Build the final gold answers dictionary
gold_answers = {}
missing = 0

for sample in formatted_samples:
    research_id = str(sample["research_id"])
    matching_key = (sample["id"], sample["answerable"])

    if matching_key not in raw_lookup:
        print(f"Warning: Could not find raw match for {matching_key}")
        missing += 1
        continue

    raw = raw_lookup[matching_key]

    # Handle unanswerable cases
    if not raw["answerable"]:
        gold_answers[research_id] = ["unanswerable"]
    else:
        answers = [raw["answer"]]
        aliases = raw.get("answer_aliases", [])
        for alias in aliases:
            if alias not in answers:
                answers.append(alias)
        gold_answers[research_id] = answers

# Save to JSON
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(gold_answers, f, indent=2)

print(f"Golden answers saved to: {output_path}")
if missing > 0:
    print(f"Warning: {missing} formatted samples had no raw match.")
