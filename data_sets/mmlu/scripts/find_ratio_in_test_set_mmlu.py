import os
import csv
import math

# === FILL IN THESE PATHS ===
INPUT_FOLDER = "data_sets/mmlu/mmlu_original/test"            # Folder containing CSV files
OUTPUT_FOLDER = "data_sets/mmlu"                              # Folder to save output file
OUTPUT_FILENAME = "entry_distribution.txt"                    # Name of the output file

# === Ensure output directory exists ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Count entries per CSV file and overall ===
file_counts = {}
total_count = 0

for filename in os.listdir(INPUT_FOLDER):
    if not filename.endswith(".csv"):
        continue

    file_path = os.path.join(INPUT_FOLDER, filename)
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        count = len(reader)
        file_counts[filename] = count
        total_count += count

# === Calculate raw proportions for 500 samples ===
proportions = {
    filename: (count / total_count) * 500
    for filename, count in file_counts.items()
}

# === Convert to integers with rounding, then adjust to sum to exactly 500 ===
rounded_allocations = {k: int(v) for k, v in proportions.items()}
residuals = {k: proportions[k] - rounded_allocations[k] for k in proportions}

current_sum = sum(rounded_allocations.values())
delta = 500 - current_sum

# Sort residuals descending for upward adjustment, ascending for downward
if delta > 0:
    for k in sorted(residuals, key=residuals.get, reverse=True)[:delta]:
        rounded_allocations[k] += 1
elif delta < 0:
    for k in sorted(residuals, key=residuals.get)[:abs(delta)]:
        rounded_allocations[k] -= 1

# === Write the result to a text file ===
output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

with open(output_path, "w", encoding="utf-8") as out_file:
    out_file.write(f"Total entries across all files: {total_count}\n\n")
    out_file.write(f"Target: 500 samples\n\n")
    for filename, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        original_ratio = (count / total_count) * 100 if total_count > 0 else 0
        allocated = rounded_allocations[filename]
        out_file.write(f"{filename}: {count} entries ({original_ratio:.2f}%) -> {allocated} samples\n")

print(f"Distribution with 500-sample allocation saved to: {output_path}")
