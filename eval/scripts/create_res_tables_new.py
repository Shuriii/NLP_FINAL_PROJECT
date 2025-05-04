import os
import json
import pandas as pd
from datetime import datetime

# Model mapping (to be filled externally)
# Model mapping (reloaded after kernel reset)
model_id_map = {
    1: "gemma-2-2b",
    2: "gemma-2-2b_duplication_[(0,1),(25,1)]",
    3: "gemma-2-2b_duplication_[(12,1),(13,1)]",
    4: "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]",
    5: "gemma-2-2b_duplication_[(0,3),(22,3)]",
    6: "gemma-2-2b_duplication_[(0,4)]",
    7: "gemma-2-2b_duplication_[(10,5)]",
    8: "gemma-2-2b_duplication_[(11,3)]",
    9: "gemma-2-2b_duplication_[(13,2)]",
    10: "gemma-2-2b_duplication_[(22,4)]",
    11: "gemma-2-9b",
  #  12: "gemma-2-27b",
    13: "Meta-Llama-3-8B",
    14: "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]",
    15: "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]",
    16: "Meta-Llama-3-8B_duplication_[(0,4),(27,4)]",
    17: "Meta-Llama-3-8B_duplication_[(0,5)]",
    18: "Meta-Llama-3-8B_duplication_[(13,5)]",
    19: "Meta-Llama-3-8B_duplication_[(14,3)]",
    20: "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]",
    21: "Meta-Llama-3-8B_duplication_[(16,2)]",
    22: "Meta-Llama-3-8B_duplication_[(27,5)]",
 #   23: "Meta-Llama-3-70B",
    24: "Meta-Llama-3-8B_duplication_[(26,3),(29,3)]",
    25: "gemma-2-2b_duplication_[(24,2)]",
    26: "gemma-2-2b_duplication_[(23,1),(24,1),(25,1),(26,1),(27,1)]",
    27: "gemma-2-2b_duplication_[(22,3),(25,3)]",
    28: "gemma-2-2b_duplication_[(21,7)]",
    29: "gemma-2-2b_duplication_[(14,2),(16,2),(18,2)]",
    30: "gemma-2-2b_duplication_[(14,2),(14,2),(16,2),(16,2)]",
    31: "gemma-2-2b_duplication_[(13,1),(14,1),(15,1)]",
    32: "gemma-2-2b_duplication_[(12,3),(15,3)]",
    33: "gemma-2-2b_duplication_[(12,1),(13,1),(14,1),(15,1)]",
    34: "Meta-Llama-3-8B_duplication_[(28,2)]",
    35: "Meta-Llama-3-8B_duplication_[(27,1),(28,1),(29,1),(30,1),(31,1)]",
    36: "Meta-Llama-3-8B_duplication_[(25,7)]",
    37: "Meta-Llama-3-8B_duplication_[(16,2),(18,2),(20,2)]",
    38: "Meta-Llama-3-8B_duplication_[(16,2),(16,2),(18,2),(18,2)]",
    39: "Meta-Llama-3-8B_duplication_[(15,1),(16,1),(17,1)]",
    40: "Meta-Llama-3-8B_duplication_[(14,3),(17,3)]",
    41: "Meta-Llama-3-8B_duplication_[(14,1),(15,1),(16,1),(17,1)]"
    

}

# Model mapping (reloaded after kernel reset)


# Model configuration mapping
model_config_map = {
    "gemma-2-2b": "original",
    "gemma-2-2b_duplication_[(0,1),(25,1)]": "G",  #first, last/ in_place / small/ 
    "gemma-2-2b_duplication_[(12,1),(13,1)]": "H", #middle/ in_place / small / true
    "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]": "I", #first, middle, last/ in_place / small
    "gemma-2-2b_duplication_[(0,3),(22,3)]": "F", # first, last/ blocks / medium
    "gemma-2-2b_duplication_[(0,4)]": "D", # first/ blocks / big
    "gemma-2-2b_duplication_[(10,5)]": "C", # middle/ blocks / big
    "gemma-2-2b_duplication_[(11,3)]": "A", #middle/ blocks / medium
    "gemma-2-2b_duplication_[(13,2)]": "B", #middle/ in_place / medium
    "gemma-2-2b_duplication_[(22,4)]": "E", #last/ blocks / medium
    "gemma-2-9b": "original", 
    "Meta-Llama-3-8B": "original", 
    "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]": "I", #first, middle, last/ in_place / small
    "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]": "G", #first, last/ in_place / small
    "Meta-Llama-3-8B_duplication_[(0,4),(27,4)]": "F", # first, last/ blocks / medium
    "Meta-Llama-3-8B_duplication_[(0,5)]": "D", # first/ blocks / big
    "Meta-Llama-3-8B_duplication_[(13,5)]": "C", # middle/ blocks / big
    "Meta-Llama-3-8B_duplication_[(14,3)]": "A", #middle/ blocks / medium
    "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]": "H", #middle/ in_place / small / true
    "Meta-Llama-3-8B_duplication_[(16,2)]": "B", # middle/ in_place / medium
    "Meta-Llama-3-8B_duplication_[(27,5)]": "E", # last/ blocks / big
    "Meta-Llama-3-70B": "original",
    "Meta-Llama-3-8B_duplication_[(26,3),(29,3)]": "R",  # last / blocks / medium / true
    "gemma-2-2b_duplication_[(24,2)]": "P",              # last / blocks / small
    "gemma-2-2b_duplication_[(23,1),(24,1),(25,1),(26,1),(27,1)]": "S",  # last / in_place / small / true
    "gemma-2-2b_duplication_[(22,3),(25,3)]": "R",       # last / blocks / medium / true
    "gemma-2-2b_duplication_[(21,7)]": "Q",              # last / blocks / big
    "gemma-2-2b_duplication_[(14,2),(16,2),(18,2)]": "M",# middle / blocks / small / true
    "gemma-2-2b_duplication_[(14,2),(14,2),(16,2),(16,2)]": "O",  # middle / blocks / small / true
    "gemma-2-2b_duplication_[(13,1),(14,1),(15,1)]": "L",# middle / in_place / small / true
    "gemma-2-2b_duplication_[(12,3),(15,3)]": "N",       # middle / blocks / medium / true
    "gemma-2-2b_duplication_[(12,1),(13,1),(14,1),(15,1)]": "K",  # middle / in_place / small / true
    "Meta-Llama-3-8B_duplication_[(28,2)]": "P",         # last / blocks / small
    "Meta-Llama-3-8B_duplication_[(27,1),(28,1),(29,1),(30,1),(31,1)]": "S",  # last / in_place / small / true
    "Meta-Llama-3-8B_duplication_[(25,7)]": "Q",         # last / blocks / big
    "Meta-Llama-3-8B_duplication_[(16,2),(18,2),(20,2)]": "M",  # middle / blocks / small / true
    "Meta-Llama-3-8B_duplication_[(16,2),(16,2),(18,2),(18,2)]": "O",  # middle / blocks / small / true
    "Meta-Llama-3-8B_duplication_[(15,1),(16,1),(17,1)]": "L",  # middle / in_place / small / true
    "Meta-Llama-3-8B_duplication_[(14,3),(17,3)]": "N",  # middle / blocks / medium / true
    "Meta-Llama-3-8B_duplication_[(14,1),(15,1),(16,1),(17,1)]": "K"  # middle / in_place / small / true
}

# Configuration attribute mapping
configuration_attribute_map = {
    "A": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "medium",
        "continuous": False
    },
    "B": {
        "position": ["middle"],
        "technique": "in place",
        "size": "medium",
        "continuous": False
    },
    "C": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "big",
        "continuous": False
    },
    "D": {
        "position": ["first"],
        "technique": "blocks",
        "size": "big",
        "continuous": False
    },
    "E": {
        "position": ["last"],
        "technique": "blocks",
        "size": "medium",
        "continuous": False
    },
    "F": {
        "position": ["first", "last"],
        "technique": "blocks",
        "size": "medium",
        "continuous": False
    },
    "G": {
        "position": ["first", "last"],
        "technique": "in place",
        "size": "small",
        "continuous": False
    },
    "H": {
        "position": ["middle"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    },
    "I": {
        "position": ["first", "middle", "last"],
        "technique": "in place",
        "size": "small",
        "continuous": False
    },
    "original": {
        "position": [],
        "technique": None,
        "size": None,
        "continuous": None
    },
    "K": {
        "position": ["middle"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    },
    "L": {
        "position": ["middle"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    },
    "M": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "small",
        "continuous": True
    },
    "N": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "medium",
        "continuous": True
    },
    "O": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "small",
        "continuous": True
    },
    "P": {
        "position": ["last"],
        "technique": "blocks",
        "size": "small",
        "continuous": False
    },
    "Q": {
        "position": ["last"],
        "technique": "blocks",
        "size": "big",
        "continuous": False
    },
    "R": {
        "position": ["last"],
        "technique": "blocks",
        "size": "medium",
        "continuous": True
    },
    "S": {
        "position": ["last"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    }
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
