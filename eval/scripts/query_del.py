# Updated implementation with guaranteed structure even when no entries are found

import os
import pandas as pd
from datetime import datetime

# Number of recent files to keep for each dataset
DATASET_FILE_COUNTS = {
    "mmlu": 10,
    "drop": 6,
    "musique": 12
}

# Static input/output directories
INPUT_DIR = "eval/summary_tables"
OUTPUT_DIR = "eval/analysis"

def get_latest_files(dir_path, count):
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)), reverse=True)
    return files[:count]

def parse_filename(file_name):
    parts = file_name.replace(".csv", "").split("_")
    dataset = parts[0] if len(parts) > 0 else "unknown"
    family = parts[1] if len(parts) > 1 else "unknown"

    # Detect if there's a super category or just timestamp
    if len(parts) <= 3:
        super_category = "general"
    else:
        super_category_parts = []
        for i in range(2, len(parts) - 2):  # Skip last two parts which are timestamp
            super_category_parts.append(parts[i])
        super_category = " ".join(super_category_parts) if super_category_parts else "general"

    return dataset, family, super_category

def analyze_query_1():
    model = "gemma-2-9b"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"query_1_{timestamp}.csv")
    matching_files = []

    for dataset, count in DATASET_FILE_COUNTS.items():
        dataset_dir = os.path.join(INPUT_DIR, dataset)
        if not os.path.exists(dataset_dir):
            continue

        recent_files = get_latest_files(dataset_dir, count)

        for file in recent_files:
            file_path = os.path.join(dataset_dir, file)
            try:
                df = pd.read_csv(file_path)

                column = None
                for candidate in ["accuracy", "average_f1"]:
                    if candidate in df.columns:
                        column = candidate
                        break

                if column is None:
                    continue

                if model not in df["model"].values:
                    continue

                top_model = df.sort_values(by=column, ascending=False).iloc[0]["model"]
                if top_model != model:
                    base_name = "_".join(file.split("_")[:-1]) + ".csv"
                    matching_files.append(base_name)
            except Exception:
                continue

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"filename": matching_files}).to_csv(output_file, index=False)
    print(f"Query 1 completed. Results saved to: {output_file}")

def analyze_query_2():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_entries = []

    for dataset, count in DATASET_FILE_COUNTS.items():
        dataset_dir = os.path.join(INPUT_DIR, dataset)
        if not os.path.exists(dataset_dir):
            continue

        recent_files = get_latest_files(dataset_dir, count)

        for file in recent_files:
            file_path = os.path.join(dataset_dir, file)
            try:
                df = pd.read_csv(file_path)
                if "configuration" not in df.columns:
                    continue

                df = df.reset_index(drop=True)
                for i in reversed(range(len(df))):
                    if str(df.at[i, "configuration"]).lower() == "original":
                        break
                else:
                    continue  # No original found

                for j in range(0, i):
                    if str(df.at[j, "configuration"]).lower() == "original":
                        continue
                    row = df.iloc[j]
                    dataset_name, family, super_category = parse_filename(file)
                    all_entries.append({
                        "configuration": row["configuration"],
                        "model": row["model"],
                        "family": family,
                        "dataset": dataset_name,
                        "super_category": super_category,
                        "% improvement": row.get("% improvement", ""),
                        "source_file": file
                    })
            except Exception:
                continue

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_df = pd.DataFrame(all_entries, columns=[
        "configuration", "model", "family", "dataset", "super_category", "% improvement", "source_file"
    ])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_file = os.path.join(OUTPUT_DIR, f"query_2_{timestamp}.csv")
    full_df.to_csv(main_file, index=False)

    # Save filtered tables by family
    for family in ["llama", "gemma"]:
        family_df = full_df[full_df["family"] == family]
        family_file = os.path.join(OUTPUT_DIR, f"query_2_{family}_{timestamp}.csv")
        family_df.to_csv(family_file, index=False)

    # Save filtered tables by super category
    for super_cat in full_df["super_category"].unique().tolist():
        super_df = full_df[full_df["super_category"] == super_cat]
        super_file = os.path.join(OUTPUT_DIR, f"query_2_{super_cat}_{timestamp}.csv")
        super_df.to_csv(super_file, index=False)

    print(f"Query 2 completed. Results saved to: {main_file}")

def analyze_query_3():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows = []

    required_columns = [
        "model", "configuration", "position",
        "technique", "size", "continuous", "% improvement"
    ]

    def parse_filename_parts(file_name):
        parts = file_name.replace(".csv", "").split("_")
        dataset = parts[0] if len(parts) > 0 else "unknown"
        family = parts[1] if len(parts) > 1 else "unknown"

        # Assume last two parts are timestamp
        core_parts = parts[:-2]
        if len(core_parts) <= 2:
            super_category = "general"
        else:
            super_category = " ".join(core_parts[2:])  # join all between family and timestamp

        return dataset, family, super_category

    for dataset, count in DATASET_FILE_COUNTS.items():
        dataset_dir = os.path.join(INPUT_DIR, dataset)
        if not os.path.exists(dataset_dir):
            continue

        recent_files = get_latest_files(dataset_dir, count)

        for file in recent_files:
            file_path = os.path.join(dataset_dir, file)
            try:
                df = pd.read_csv(file_path)

                if all(col in df.columns for col in required_columns):
                    filtered_df = df[required_columns].copy()
                    dataset_name, family, super_category = parse_filename_parts(file)
                    filtered_df["dataset"] = dataset_name
                    filtered_df["family"] = family
                    filtered_df["super_category"] = super_category
                    all_rows.append(filtered_df)
            except Exception:
                continue

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"query_3_{timestamp}.csv")

    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=required_columns + ["dataset", "family", "super_category"])

    combined_df.to_csv(output_file, index=False)
    print(f"Query 3 completed. Results saved to: {output_file}")


def run_query(query_id):
    if query_id == 1:
        analyze_query_1()
    elif query_id == 2:
        analyze_query_2()
    elif query_id == 3:
        analyze_query_3()
    else:
        print(f"Query ID {query_id} not implemented.")

# Activate here
run_query(3)
