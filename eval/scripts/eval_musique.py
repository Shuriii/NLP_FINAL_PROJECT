import string
from datetime import datetime
import json
import os

# Re-define model ID to name mapping after kernel reset
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
   # 12: "gemma-2-27b",
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
   # 23: "Meta-Llama-3-70B",
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

