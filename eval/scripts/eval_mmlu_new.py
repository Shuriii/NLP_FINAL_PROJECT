from datetime import datetime
import json
import os
from collections import defaultdict

# Model mapping file
# This file contains the mapping of model IDs to their names and configurations
maps_path = "eval/scripts/model_registry.json"

# Load the maps
with open(maps_path, "r") as f:
    maps = json.load(f)

model_id_map = {int(k): v for k, v in maps["model_id_map"].items()}


def evaluate_accuracy(preds_path, gold_path):
    with open(preds_path, "r") as f:
        preds = json.load(f)
    with open(gold_path, "r") as f:
        gold = json.load(f)

    correct = 0
    total = 0

    # Track per-subject and per-super-category stats
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    supercat_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for k, v in gold.items():
        gold_answer = v["answer"]
        subject = v["subject"]
        supercat = v["super_category"]

        if k in preds:
            total += 1
            subject_stats[subject]["total"] += 1
            supercat_stats[supercat]["total"] += 1

            if preds[k].strip() == gold_answer.strip():
                correct += 1
                subject_stats[subject]["correct"] += 1
                supercat_stats[supercat]["correct"] += 1

    accuracy = correct / total if total else 0

    # Format subject-wise accuracy
    per_subject = {}
    for subj, stats in subject_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        per_subject[subj] = {
            "accuracy": acc,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy_percent": round(acc * 100, 2)
        }

    # Format super-category accuracy
    per_super_category = {}
    for cat, stats in supercat_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        per_super_category[cat] = {
            "accuracy": acc,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy_percent": round(acc * 100, 2)
        }

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "accuracy_percent": round(accuracy * 100, 2),
        "per_subject": per_subject,
        "per_super_category": per_super_category
    }

def evaluate_models(model_ids):
    results = {}
    gold_path = "eval/golden_ans/mmlu_gold_ans_new.json"
    for mid in model_ids:
        model_name = model_id_map[mid]
        pred_path = f"eval/predictions/{model_name}/mmlu/predictions.json"

        result = evaluate_accuracy(pred_path, gold_path)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"mmlu_{model_name}_{now}.json"
        result_dir = f"eval/results/{model_name}/mmlu"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, result_filename), "w") as f:
            json.dump(result, f, indent=2)

        results[model_name] = result

        # Also print results to screen
        #print(f"\n===== {model_name} =====")
        #print(json.dumps(result, indent=2))

    return results

# Example usage
evaluate_models(list(model_id_map.keys()))

print("Evaluation completed.")
