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

def model_not_ranked_first(df, column, model):
    if model not in df['model'].values:
        return False  # Model not in file, skip
    sorted_df = df.sort_values(by=column, ascending=False)
    top_model = sorted_df.iloc[0]['model']
    return top_model != model

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
                if model_not_ranked_first(df, column, model):
                    base_name = "_".join(file.split("_")[:-1]) + ".csv"
                    matching_files.append(base_name)
            except Exception:
                continue

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"filename": matching_files}).to_csv(output_file, index=False)
    print(f"Query 1 completed. Results saved to: {output_file}")

def extract_metadata_from_filename(file_name):
    parts = file_name.split("_")
    dataset = parts[0]
    family = parts[1]
    super_category = parts[2] if len(parts) > 3 else "general"
    return dataset, family, super_category

def analyze_query_2():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"query_2_{timestamp}.csv")
    results = []

    for dataset, count in DATASET_FILE_COUNTS.items():
        dataset_dir = os.path.join(INPUT_DIR, dataset)
        if not os.path.exists(dataset_dir):
            continue

        recent_files = get_latest_files(dataset_dir, count)

        for file in recent_files:
            file_path = os.path.join(dataset_dir, file)
            try:
                df = pd.read_csv(file_path)
                df = df.reset_index(drop=True)
                if 'configuration' not in df.columns:
                    continue

                original_indices = df.index[df['configuration'] == 'original'].tolist()
                if not original_indices:
                    continue

                top_index = min(original_indices)
                top_configs = df.iloc[:top_index]
                dataset_name, family, super_category = extract_metadata_from_filename(file)

                for _, row in top_configs.iterrows():
                    results.append({
                        "configuration": row.get("configuration", ""),
                        "family": family,
                        "dataset": dataset_name,
                        "super_category": super_category,
                        "position": row.get("position", ""),
                        "technique": row.get("technique", ""),
                        "size": row.get("size", ""),
                        "continuous": row.get("continuous", ""),
                        "score": row.get("accuracy_percent", row.get("average_f1_percent", "")),
                        "% improvement": row.get("% improvement", ""),
                        "source_file": file
                    })
            except Exception:
                continue

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out = pd.DataFrame(results)
    if "configuration" in df_out.columns:
        df_out = df_out.sort_values(by="configuration")
    df_out.to_csv(output_file, index=False)
    print(f"Query 2 completed. Results saved to: {output_file}")

# === Query Dispatcher ===
def run_query(query_id):
    if query_id == 1:
        analyze_query_1()
    elif query_id == 2:
        analyze_query_2()
    else:
        print(f"Query ID {query_id} not implemented.")

# === Trigger execution here ===
run_query(2)
