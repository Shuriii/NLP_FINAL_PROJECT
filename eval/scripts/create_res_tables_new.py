import os
import json
import pandas as pd
from datetime import datetime

# Model mapping file
# This file contains the mapping of model IDs to their names and configurations
maps_path = "eval/scripts/model_registry.json"

# Load the maps
with open(maps_path, "r") as f:
    maps = json.load(f)

model_id_map = {int(k): v for k, v in maps["model_id_map"].items()}
model_config_map = maps["model_config_map"]
configuration_attribute_map = maps["configuration_attribute_map"]

def get_latest_result_file(path):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith('.json')]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
    return os.path.join(path, latest)

def compute_improvement(df, baseline_name, score_col):
    if baseline_name not in df['model'].values:
        df['% improvement'] = ""
        return df
    baseline_score = df[df['model'] == baseline_name][score_col].values[0]
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

        for df_rows, baseline_id, label in [(llama_rows, 13, "llama"), (gemma_rows, 1, "gemma")]:
            if not df_rows:
                continue
            df = pd.DataFrame(df_rows)
            if df.empty:
                continue

            score_col = determine_score_column(df, dataset)
            baseline_name = model_id_map.get(baseline_id, df['model'].iloc[0])
            df.sort_values(by=score_col, ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.insert(0, 'configuration', df['model'].map(model_config_map))

            # Add configuration attributes
            df['position'] = df['configuration'].map(lambda x: ', '.join(configuration_attribute_map.get(x, {}).get('position', [])))
            df['technique'] = df['configuration'].map(lambda x: configuration_attribute_map.get(x, {}).get('technique'))
            df['size'] = df['configuration'].map(lambda x: configuration_attribute_map.get(x, {}).get('size'))
            df['continuous'] = df['configuration'].map(lambda x: configuration_attribute_map.get(x, {}).get('continuous'))

            df = compute_improvement(df, baseline_name, score_col)

            columns_to_keep = ['model', 'configuration', 'position', 'technique', 'size', 'continuous', score_col, 'total', 'correct', 'accuracy_percent', 'average_f1', 'average_f1_percent', 'total_evaluated', '% improvement']
            df = df[[col for col in columns_to_keep if col in df.columns]]

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = f"eval/summary_tables/{dataset}"
            os.makedirs(output_folder, exist_ok=True)

            general_path = os.path.join(output_folder, f"{dataset}_{label}_{now}.csv")
            df.to_csv(general_path, index=False)

            if dataset in ["mmlu", "drop"] and 'per_super_category' in df_rows[0]:
                supercats = df_rows[0]['per_super_category'].keys()
                for supercat in supercats:
                    supercat_rows = []
                    for row in df_rows:
                        stats = row.get('per_super_category', {}).get(supercat)
                        if stats:
                            config_letter = model_config_map.get(row["model"], "")
                            attr = configuration_attribute_map.get(config_letter, {})
                            if dataset == "mmlu":
                                supercat_rows.append({
                                    "model": row["model"],
                                    "configuration": config_letter,
                                    "position": ', '.join(attr.get("position", [])),
                                    "technique": attr.get("technique"),
                                    "size": attr.get("size"),
                                    "continuous": attr.get("continuous"),
                                    "accuracy": stats["accuracy"],
                                    "total": stats["total"],
                                    "correct": stats["correct"],
                                    "accuracy_percent": stats["accuracy_percent"]
                                })
                            elif dataset == "drop":
                                supercat_rows.append({
                                    "model": row["model"],
                                    "configuration": config_letter,
                                    "position": ', '.join(attr.get("position", [])),
                                    "technique": attr.get("technique"),
                                    "size": attr.get("size"),
                                    "continuous": attr.get("continuous"),
                                    "average_f1": stats["average_f1"],
                                    "average_f1_percent": stats["average_f1_percent"],
                                    "total_evaluated": stats["total_evaluated"]
                                })
                    if supercat_rows:
                        supercat_df = pd.DataFrame(supercat_rows)
                        score_col = "accuracy" if dataset == "mmlu" else "average_f1"
                        supercat_df.sort_values(by=score_col, ascending=False, inplace=True)
                        supercat_df.reset_index(drop=True, inplace=True)
                        supercat_df = compute_improvement(supercat_df, baseline_name, score_col)
                        supercat_path = os.path.join(output_folder, f"{dataset}_{label}_{supercat}_{now}.csv")
                        supercat_df.to_csv(supercat_path, index=False)

# Example usage:
summarize_results("all")
print("Summary tables created successfully.")
