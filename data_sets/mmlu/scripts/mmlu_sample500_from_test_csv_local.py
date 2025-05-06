import os
import json
import random
import csv

# === FILL IN THESE PATHS ===
INPUT_CSV_FOLDER = "data_sets/mmlu/mmlu_original/test"  # Folder containing files like professional_law_test.csv
OUTPUT_FOLDER = "data_sets/mmlu/mmlu_raw_data_ samples"  # Folder to save the output JSON file
OUTPUT_FILENAME = "mmlu_500_samples.json"  # Output file name

# === Fixed number of samples to draw per topic ===
topic_sample_counts = {
    'professional_law': 55, 'moral_scenarios': 32, 'miscellaneous': 28, 'professional_psychology': 22,
    'high_school_psychology': 19, 'high_school_macroeconomics': 14, 'elementary_mathematics': 13,
    'moral_disputes': 12, 'prehistory': 11, 'philosophy': 11, 'high_school_biology': 11,
    'nutrition': 11, 'professional_accounting': 10, 'professional_medicine': 10,
    'high_school_mathematics': 10, 'clinical_knowledge': 9, 'security_studies': 9,
    'high_school_microeconomics': 8, 'high_school_world_history': 8, 'conceptual_physics': 8,
    'marketing': 8, 'human_aging': 8, 'high_school_statistics': 8,
    'high_school_us_history': 7, 'high_school_chemistry': 7, 'sociology': 7,
    'high_school_geography': 7, 'high_school_government_and_politics': 7,
    'college_medicine': 6, 'world_religions': 6, 'virology': 6, 'high_school_european_history': 6,
    'logical_fallacies': 6, 'astronomy': 5, 'high_school_physics': 5,
    'electrical_engineering': 5, 'college_biology': 5, 'anatomy': 5,
    'human_sexuality': 5, 'formal_logic': 4, 'international_law': 4, 'econometrics': 4,
    'machine_learning': 4, 'public_relations': 4, 'jurisprudence': 4,
    'management': 4, 'college_physics': 4, 'us_foreign_policy': 4, 'global_facts': 4,
    'business_ethics': 4, 'abstract_algebra': 4, 'medical_genetics': 4,
    'high_school_computer_science': 4, 'college_chemistry': 4,
    'college_computer_science': 4, 'college_mathematics': 3, 'computer_security': 3
}

# === Set fixed seed for reproducibility ===
random.seed(42)

# === Collect samples ===
llmu_samples = []

for topic, num_samples in topic_sample_counts.items():
    print(f"Sampling {num_samples} examples from topic: {topic}")
    csv_file_path = os.path.join(INPUT_CSV_FOLDER, f"{topic}_test.csv")

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found for topic: {topic} at {csv_file_path}")

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile))
        if len(reader) < num_samples:
            raise ValueError(f"Topic {topic} does not have enough examples ({len(reader)} available)")

        sampled_rows = random.sample(reader, num_samples)

        for row in sampled_rows:
            question = row[0].strip()
            choices = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
            answer = row[5].strip().upper()  # e.g., "C"
            llmu_samples.append({
                "question": question,
                "choices": choices,
                "answer": ["A", "B", "C", "D"].index(answer),
                "subject": topic
            })

# === Save final JSON ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(llmu_samples, f, indent=2)

print(f"\n Saved {len(llmu_samples)} MMLU samples to: {output_path}")
