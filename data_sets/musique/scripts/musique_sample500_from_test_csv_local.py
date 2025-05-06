import json
import random
from datasets import load_dataset

# Set fixed seed for reproducibility
random.seed(42)

# -----------------------------------
# Load MuSiQue from your local JSONL
# -----------------------------------

# CHANGE THIS PATH to your real file location  
musique_path = "data_sets/musique/original_musique_downloaded/musique_full_v1.0_dev.jsonl"

# Load MuSiQue dev set manually
with open(musique_path, "r") as f:
    musique_dataset = [json.loads(line) for line in f]

# Split answerable and unanswerable
answerable = [item for item in musique_dataset if item["answerable"]]
unanswerable = [item for item in musique_dataset if not item["answerable"]]

# Sample 400 answerable + 100 unanswerable
musique_samples = random.sample(answerable, 300) + random.sample(unanswerable, 200)

# -------- Specify your output path here --------
output_path = "data_sets/musique/musique_raw_data_ samples/musique_500_samples_from_dev.json"
# -----------------------------------------------

# Save MuSiQue 500 samples
with open(output_path, "w") as f:
    json.dump(musique_samples, f, indent=2)

print(f"Saved MuSiQue 500 samples to: {output_path}")
