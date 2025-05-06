import os
import json
import csv

# === FILL IN THESE PATHS ===
INPUT_CSV_FOLDER = "data_sets/mmlu/mmlu_original/val"   # Folder with files like professional_law_dev.csv
OUTPUT_FOLDER = "data_sets/mmlu/mmlu_5shot_samples"      # Folder to save subject_5shot.json files

# === Ensure output directory exists ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Iterate over all _dev.csv files in input folder ===
for filename in os.listdir(INPUT_CSV_FOLDER):
    if not filename.endswith("_val.csv"):
        continue

    topic = filename.replace("_val.csv", "")
    csv_path = os.path.join(INPUT_CSV_FOLDER, filename)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile))

        # Rank by total text length: question + choices
        scored_rows = []
        for row in reader:
            question = row[0].strip()
            choices = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
            total_length = len(question) + sum(len(c) for c in choices)
            scored_rows.append((total_length, row))

        # Sort by length and take 5 shortest
        top_five = sorted(scored_rows, key=lambda x: x[0])[:5]

        examples = []
        for _, row in top_five:
            question = row[0].strip()
            choices = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
            answer = row[5].strip().upper()
            examples.append({
                "question": question,
                "choices": choices,
                "answer": ["A", "B", "C", "D"].index(answer),
                "subject": topic
            })

    # Save per-subject 5-shot file
    output_path = os.path.join(OUTPUT_FOLDER, f"{topic}_5shot.json")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(examples, f, indent=2)

    print(f"Saved 5-shot examples for {topic} to: {output_path}")
