import os
import json
from datetime import datetime
from collections import defaultdict

# Model mapping file
maps_path = "eval/scripts/model_registry.json"
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
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    supercat_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    correct_ids = {"all": []}
    
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

                correct_ids["all"].append(k)
                if supercat not in correct_ids:
                    correct_ids[supercat] = []
                correct_ids[supercat].append(k)

    accuracy = correct / total if total else 0

    per_subject = {}
    for subj, stats in subject_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        per_subject[subj] = {
            "accuracy": acc,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy_percent": round(acc * 100, 2)
        }

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
        "per_super_category": per_super_category,
        "correct_ids": correct_ids
    }


def evaluate_models(model_ids):
    results = {}
    gold_path = "eval/golden_ans/mmlu_gold_ans_new.json"

    for mid in model_ids:
        model_name = model_id_map[mid]
        pred_path = f"eval/predictions/{model_name}/mmlu/predictions.json"

        result = evaluate_accuracy(pred_path, gold_path)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main results
        result_dir = f"eval/results/{model_name}/mmlu"
        os.makedirs(result_dir, exist_ok=True)
        result_filename = f"mmlu_{model_name}_{now}.json"
        with open(os.path.join(result_dir, result_filename), "w") as f:
            json.dump({k: v for k, v in result.items() if k != "correct_ids"}, f, indent=2)

        # Save correct IDs
        id_dir = f"eval/correct_ans_ids/{model_name}/mmlu"
        os.makedirs(id_dir, exist_ok=True)
        id_filename = f"mmlu_{model_name}_correct_ids_{now}.json"
        with open(os.path.join(id_dir, id_filename), "w") as f:
            json.dump(result["correct_ids"], f, indent=2)

        results[model_name] = result

    return results


evaluate_models(list(model_id_map.keys()))
print("Evaluation completed.")
