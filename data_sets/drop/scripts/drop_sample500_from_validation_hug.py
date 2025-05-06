import os
import json
import random
from datasets import load_dataset

# === Fixed paths ===
OUTPUT_FOLDER = "data_sets/drop/drop_raw_data_samples"
OUTPUT_FILENAME = "drop_500_samples.json"

# === Set fixed seed for reproducibility ===
random.seed(42)

# --- Load DROP from Hugging Face ---
drop_dataset = load_dataset("drop", split="validation")

# Split by section type
history_samples_all = [x for x in drop_dataset if x["section_id"].startswith("history_")]
nfl_samples_all = [x for x in drop_dataset if x["section_id"].startswith("nfl_")]

# Sample exact amounts
history_samples = random.sample(history_samples_all, 350)
nfl_samples = random.sample(nfl_samples_all, 150)

# Combine
drop_samples = history_samples + nfl_samples

# Save final JSON
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(drop_samples, f, indent=2)

print(f"\nSaved {len(drop_samples)} DROP samples to: {output_path}")
print("DROP 500 samples saved with 350 history and 150 NFL!")
