import os
import json
import pandas as pd
from datetime import datetime

# Model mapping (reloaded after kernel reset)
model_id_map = {
    1: "gemma-2-2b",
    2: "gemma-2-2b_duplication_[(0,1),(25,1)]",
    3: "gemma-2-2b_duplication_[(12,1),(13,1)]",
    4: "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]",
    5: "gemma-2-9b",
    6: "gemma-2-27b",
    7: "Meta-Llama-3-8B",
    8: "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]",
    9: "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]",
    10: "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]",
    11: "Meta-Llama-3-70B"
}

def get_latest_result_file(path):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
    return os.path.join(path, latest)

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

        llama_df.insert(0, '', '')
        gemma_df.insert(0, '', '')

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"eval/summary_tables/{dataset}"
        os.makedirs(output_folder, exist_ok=True)

        llama_path = os.path.join(output_folder, f"{dataset}_llama_{now}.csv")
        gemma_path = os.path.join(output_folder, f"{dataset}_gemma_{now}.csv")

        llama_df.to_csv(llama_path, index=False)
        gemma_df.to_csv(gemma_path, index=False)

    return "Summary tables generated."

# Example usage:
summarize_results(["mmlu", "musique"])
# summarize_results(["drop", "musique"])
# summarize_results("all")

