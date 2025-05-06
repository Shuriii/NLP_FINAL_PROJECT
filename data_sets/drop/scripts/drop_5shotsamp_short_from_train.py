import os
import json
from datasets import load_dataset

# === FILL IN THIS OUTPUT PATH ===
OUTPUT_FOLDER = "data_sets/drop/drop_5shot_samples"
# =================================

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load DROP training set
drop_train = load_dataset("drop", split="train")

# Split into history and nfl entries
history_entries = []
nfl_entries = []

for sample in drop_train:
    section_id = sample.get("section_id", "")
    passage = sample.get("passage", "")
    question = sample.get("question", "")
    total_length = len(passage) + len(question)

    entry = {
        "section_id": section_id,
        "query_id": sample.get("query_id", ""),
        "passage": passage,
        "question": question,
        "answers_spans": sample.get("answers_spans", {}),
        "total_length": total_length
    }

    if section_id.startswith("history_"):
        history_entries.append(entry)
    elif section_id.startswith("nfl_"):
        nfl_entries.append(entry)

# Sort by total character length (passage + question)
history_sorted = sorted(history_entries, key=lambda x: x["total_length"])[:5]
nfl_sorted = sorted(nfl_entries, key=lambda x: x["total_length"])[:5]

# Remove helper field before saving
for entry in history_sorted + nfl_sorted:
    entry.pop("total_length", None)

# Save history samples
with open(os.path.join(OUTPUT_FOLDER, "history_5shot.json"), "w", encoding="utf-8") as f:
    json.dump(history_sorted, f, indent=2)

# Save NFL samples
with open(os.path.join(OUTPUT_FOLDER, "nfl_5shot.json"), "w", encoding="utf-8") as f:
    json.dump(nfl_sorted, f, indent=2)

print("Saved 5-shot shortest history examples to: history_5shot.json")
print("Saved 5-shot shortest NFL examples to: nfl_5shot.json")
print("5-shot samples saved successfully!")