import json
import os
import string
from datetime import datetime
from collections import defaultdict

# ===> Paths to required files
maps_path = "eval/scripts/model_registry.json"
metadata_path = "eval/golden_ans/musique_reformatted_samples.json"
gold_path = "eval/golden_ans/musique_gold_ans.json"

# ===> Load model ID map
with open(maps_path, "r") as f:
    maps = json.load(f)
model_id_map = {int(k): v for k, v in maps["model_id_map"].items()}

# ===> Load metadata file (for hop type and answerable)
with open(metadata_path, "r") as f:
    metadata = json.load(f)
metadata_lookup = {
    str(entry["research_id"]): {
        "hop": entry["id"].split("__")[0].rstrip("1234567890"),  # remove trailing digits from hop
        "answerable": "answerable" if entry.get("answerable") else "unanswerable"
    }
    for entry in metadata
}

def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = text.split("\n")[0]
    return text.translate(str.maketrans('', '', string.punctuation))

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
    per_category_scores = defaultdict(lambda: {"f1_sum": 0, "count": 0})
    correct_ids = defaultdict(list)

    for k, gold_answers in gold.items():
        if k not in preds:
            continue
        pred = normalize(preds[k])
        max_f1 = max(f1_score(pred, g) for g in gold_answers)
        total_f1 += max_f1
        total += 1

        # Track correct predictions
        #print (max_f1)
        if max_f1 == 1.0:
            correct_ids["general"].append(k)

        # Track by category
        meta = metadata_lookup.get(k)
        if meta:
            per_category_scores[meta["hop"]]["f1_sum"] += max_f1
            per_category_scores[meta["hop"]]["count"] += 1
            per_category_scores[meta["answerable"]]["f1_sum"] += max_f1
            per_category_scores[meta["answerable"]]["count"] += 1
            if max_f1 == 1.0:
                correct_ids[meta["hop"]].append(k)
                correct_ids[meta["answerable"]].append(k)

    average_f1 = total_f1 / total if total else 0
    per_super_category = {
        cat: {
            "average_f1": scores["f1_sum"] / scores["count"] if scores["count"] else 0,
            "average_f1_percent": round((scores["f1_sum"] / scores["count"]) * 100, 2) if scores["count"] else 0,
            "total_evaluated": scores["count"]
        }
        for cat, scores in per_category_scores.items()
    }

    return {
        "average_f1": average_f1,
        "average_f1_percent": round(average_f1 * 100, 2),
        "total_evaluated": total,
        "per_super_category": per_super_category,
        "correct_ids": correct_ids  # Include correct IDs in return
    }

def evaluate_models(model_ids):
    results = {}
    for mid in model_ids:
        model_name = model_id_map[mid]
        pred_path = f"eval/predictions/{model_name}/musique/predictions.json"
        result = evaluate_f1(pred_path, gold_path)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"musique_{model_name}_{now}.json"
        result_dir = f"eval/results/{model_name}/musique"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, result_filename), "w") as f:
            json.dump({k: v for k, v in result.items() if k != "correct_ids"}, f, indent=2)

        # Save correct IDs separately
        correct_ids_dir = f"eval/correct_ans_ids/{model_name}/musique"
        os.makedirs(correct_ids_dir, exist_ok=True)
        correct_ids_filename = f"musique_{model_name}_correct_ids_{now}.json"
        with open(os.path.join(correct_ids_dir, correct_ids_filename), "w") as f:
            json.dump(result["correct_ids"], f, indent=2)

        results[model_name] = result
    return results

# Run evaluation
evaluate_models(list(model_id_map.keys()))
#evaluate_models([11])  # Example model IDs
print("MuSiQue F1 evaluation complete with hop and answerability breakdown and correct ID tracking.")
