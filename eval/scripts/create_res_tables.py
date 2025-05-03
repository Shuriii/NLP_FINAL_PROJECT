import os
import json
import pandas as pd
from datetime import datetime

# Model mapping (reloaded after kernel reset)
model_id_map = {
    1: "gemma-2-2b", 2: "gemma-2-2b_duplication_[(0,1),(25,1)]", 3: "gemma-2-2b_duplication_[(12,1),(13,1)]",
    4: "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]", 5: "gemma-2-2b_duplication_[(0,3),(22,3)]",
    6: "gemma-2-2b_duplication_[(0,4)]", 7: "gemma-2-2b_duplication_[(10,5)]", 8: "gemma-2-2b_duplication_[(11,3)]",
    9: "gemma-2-2b_duplication_[(13,2)]", 10: "gemma-2-2b_duplication_[(22,4)]", 11: "gemma-2-9b",
    12: "gemma-2-27b", 13: "Meta-Llama-3-8B", 14: "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]",
    15: "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]", 16: "Meta-Llama-3-8B_duplication_[(0,4),(27,4)]",
    17: "Meta-Llama-3-8B_duplication_[(0,5)]", 18: "Meta-Llama-3-8B_duplication_[(13,5)]",
    19: "Meta-Llama-3-8B_duplication_[(14,3)]", 20: "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]",
    21: "Meta-Llama-3-8B_duplication_[(16,2)]", 22: "Meta-Llama-3-8B_duplication_[(27,5)]", 23: "Meta-Llama-3-70B"
}

# Model configuration mapping
model_config_map = {
    "gemma-2-2b": "original",
      "gemma-2-2b_duplication_[(0,1),(25,1)]": "G", 
    "gemma-2-2b_duplication_[(12,1),(13,1)]": "H",
      "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]": "I",
    "gemma-2-2b_duplication_[(0,3),(22,3)]": "F", #
      "gemma-2-2b_duplication_[(0,4)]": "D", #
    "gemma-2-2b_duplication_[(10,5)]": "C", #
      "gemma-2-2b_duplication_[(11,3)]": "A", #
    "gemma-2-2b_duplication_[(13,2)]": "B", #
      "gemma-2-2b_duplication_[(22,4)]": "E", #
    "gemma-2-9b": "original",
      "gemma-2-27b": "original", "Meta-Llama-3-8B": "original",
    "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]": "I",
      "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]": "G",
    "Meta-Llama-3-8B_duplication_[(0,4),(27,4)]": "F", #
       "Meta-Llama-3-8B_duplication_[(0,5)]": "D", #
    "Meta-Llama-3-8B_duplication_[(13,5)]": "C", #
      "Meta-Llama-3-8B_duplication_[(14,3)]": "A",
    "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]": "H", 
    "Meta-Llama-3-8B_duplication_[(16,2)]": "B", #
    "Meta-Llama-3-8B_duplication_[(27,5)]": "E", #
      "Meta-Llama-3-70B": "original"
}

def get_latest_result_file(path):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
    return os.path.join(path, latest)

def compute_improvement(df, baseline_name, score_col):
    baseline_score = df[df['model'] == baseline_name][score_col].values
    if len(baseline_score) == 0:
        return df
    baseline_score = baseline_score[0]
    df['% improvement'] = ((df[score_col] - baseline_score) / baseline_score * 100).round(2)
    return df

def determine_score_column(df, dataset):
    if dataset == "musique":
        return "average_f1"
    elif dataset == "drop":
        return "average_f1_percent"
    elif dataset == "mmlu":
        return "accuracy"
    else:
        return df.columns[-1]

def summarize_results(datasets):
    all_datasets = ['mmlu', 'drop', 'musique']
    if datasets == "all":
        datasets = all_datasets
    else:
        datasets = [ds for ds in datasets if ds in all_datasets]

    for dataset in datasets:
        llama_rows = []
        gemma_rows = []

        for model_id, model_name in model_id_map.items():
            result_dir = f"eval/results/{model_name}/{dataset}"
            result_file = get_latest_result_file(result_dir)
            if result_file:
                with open(result_file, "r") as f:
                    metrics = json.load(f)
                row = {'model': model_name}
                row.update(metrics)
                if "gemma" in model_name:
                    gemma_rows.append(row)
                else:
                    llama_rows.append(row)

        llama_df = pd.DataFrame(llama_rows)
        gemma_df = pd.DataFrame(gemma_rows)

        for df, baseline_id in [(llama_df, 13), (gemma_df, 1)]:
            if not df.empty:
                score_col = determine_score_column(df, dataset)
                baseline_name = model_id_map[baseline_id]
                df.sort_values(by=score_col, ascending=False, inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.insert(0, 'configuration', df['model'].map(model_config_map))
                df = compute_improvement(df, baseline_name, score_col)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"eval/summary_tables/{dataset}"
        os.makedirs(output_folder, exist_ok=True)

        llama_path = os.path.join(output_folder, f"{dataset}_llama_{now}.csv")
        gemma_path = os.path.join(output_folder, f"{dataset}_gemma_{now}.csv")

        llama_df.to_csv(llama_path, index=False)
        gemma_df.to_csv(gemma_path, index=False)


# Example usage:
summarize_results(["mmlu", "musique"])
# summarize_results(["drop", "musique"])
# summarize_results("all")
print("Summary tables created successfully.")