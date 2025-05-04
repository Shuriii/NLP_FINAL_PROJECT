import string
from datetime import datetime
import json
import os

# Model mapping file
# This file contains the mapping of model IDs to their names and configurations
maps_path = "eval/scripts/model_registry.json"

# Load the maps
with open(maps_path, "r") as f:
    maps = json.load(f)

model_id_map = {int(k): v for k, v in maps["model_id_map"].items()}



def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = text.split("\n")[0]
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def f1_score(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate_f1(preds_path, gold_path):
    with open(preds_path, "r") as f:
        preds = json.load(f)
    with open(gold_path, "r") as f:
        gold = json.load(f)

    total_f1 = 0
    total = 0

    for k, gold_answers in gold.items():
        if k not in preds:
            continue
        pred = normalize(preds[k])
        max_f1 = max(f1_score(pred, g) for g in gold_answers)
        total_f1 += max_f1
        total += 1

    average_f1 = total_f1 / total if total else 0
    return {
        "average_f1": average_f1,
        "average_f1_percent": round(average_f1 * 100, 2),
        "total_evaluated": total
    }

def evaluate_models(model_ids):
    results = {}
    gold_path = "eval/golden_ans/musique_gold_ans.json"
    for mid in model_ids:
        model_name = model_id_map[mid]
        pred_path = f"eval/predictions/{model_name}/musique/predictions.json"

        result = evaluate_f1(pred_path, gold_path)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"musique_{model_name}_{now}.json"
        result_dir = f"eval/results/{model_name}/musique"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, result_filename), "w") as f:
            json.dump(result, f, indent=2)

        results[model_name] = result
    return results

# Example usage
#evaluate_models([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
evaluate_models(list(model_id_map.keys()))
print("MuSiQue F1 evaluation complete.")

