from datetime import datetime
import json
import os
from collections import defaultdict

# Updated model ID to folder name mapping
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
    12: "gemma-2-27b",
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
    23: "Meta-Llama-3-70B"
}

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
evaluate_models([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
print("Evaluation completed.")
