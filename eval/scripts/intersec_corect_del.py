import os
import json
import pandas as pd
from datetime import datetime

# === CONFIGURATION SECTION (to be filled manually) ===

# Path to the root folder where each model has its correct answer JSONs
CORRECT_ANS_ROOT = "eval/correct_ans_ids"  # <-- Fill this

# List of (dataset, subcategory) pairs to examine
TARGET_PAIRS = [
   # ("musique", "4hop"),
    ("drop", "history"),
    # Add more pairs as needed
]

# Output directory for saving the result CSV
OUTPUT_DIR = "eval/analysis"  # <-- Fill this

# Models that should have gotten the sample right
models_correct = {
    "gemma-2-2b_duplication_[(10,5)]": True,
  #  "gemma-2-2b_duplication_[(22,2)]": True
    # Add more if needed
}

# Models that should have gotten the sample wrong
models_incorrect = {
   "gemma-2-2b": True,
  # "Meta-Llama-3-8B": True,
    # Add more if needed
}

# === FUNCTION LOGIC ===

def load_correct_ids(model_name, dataset, subcategory):
    model_dir = os.path.join(CORRECT_ANS_ROOT, model_name, dataset)
    if not os.path.exists(model_dir):
        return set()
    json_files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
    if not json_files:
        return set()
    json_path = os.path.join(model_dir, json_files[0])
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return set(data.get(subcategory, []))
    except Exception:
        return set()

def find_intersected_sample_ids():
    all_pair_ids = []

    for dataset, subcategory in TARGET_PAIRS:
        correct_sets = []
        for model in models_correct:
            ids = load_correct_ids(model, dataset, subcategory)
            correct_sets.append(ids)

        incorrect_sets = []
        for model in models_incorrect:
            ids = load_correct_ids(model, dataset, subcategory)
            incorrect_sets.append(ids)

        # Intersection of all correct sets
        if not correct_sets:
            return []  # If no correct data, return early
        qualified_ids = set.intersection(*correct_sets)

        # Remove any IDs from incorrect sets
        for wrong_set in incorrect_sets:
            qualified_ids.difference_update(wrong_set)

        all_pair_ids.append(qualified_ids)

    # Final intersection across all dataset-subcategory pairs
    if not all_pair_ids:
        return []
    return sorted(set.intersection(*all_pair_ids))

# === OUTPUT SECTION ===

def save_to_csv(sample_ids):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"unified_correct_{timestamp}.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    pd.DataFrame({"sample_id": sample_ids}).to_csv(output_path, index=False)
    print(f"Saved {len(sample_ids)} sample IDs to {output_path}")

# === RUN ===
if __name__ == "__main__":
    sample_ids = find_intersected_sample_ids()
    save_to_csv(sample_ids)
