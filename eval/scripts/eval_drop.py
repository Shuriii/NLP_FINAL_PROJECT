import string
from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

# Model mapping file
maps_path = "eval/scripts/model_registry.json"

with open(maps_path, "r") as f:
    maps = json.load(f)

model_id_map = {int(k): v for k, v in maps["model_id_map"].items()}

# Normalization utilities
def _remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text):
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def _remove_punc(text):
    return ''.join(ch for ch in text if ch not in EXCLUDE) if not _is_number(text) else text

def _lower(text):
    return text.lower()

def _tokenize(text):
    return re.split(" |-", text)

def _normalize_answer(text):
    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    return ' '.join(parts).strip()

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text):
    return str(float(text)) if _is_number(text) else text

def _answer_to_bags(answer):
    raw_spans = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags

def _align_bags(predicted, gold):
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores

def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    precision = intersection / float(len(predicted_bag)) if predicted_bag else 1.0
    recall = intersection / float(len(gold_bag)) if gold_bag else 1.0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set(w for w in gold_bag if _is_number(w))
    predicted_numbers = set(w for w in predicted_bag if _is_number(w))
    return not gold_numbers or bool(gold_numbers & predicted_numbers)

def f1_score(prediction, ground_truth):
    predicted_bags = _answer_to_bags(prediction)
    gold_bags = _answer_to_bags(ground_truth)
    exact_match = 1.0 if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]) else 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = round(np.mean(f1_per_bag), 2)
    return exact_match, f1

# Evaluation logic
def evaluate_f1(preds_path, gold_path):
    try:
        predicted_answers = json.load(open(preds_path, encoding='utf-8'))
        gold_annotations = json.load(open(gold_path, encoding='utf-8'))
    except Exception as e:
        print(f"Error loading files: {e}")
        return {}

    instance_exact_match = []
    instance_f1 = []
    supercat_scores = defaultdict(list)
    correct_ids_by_category = defaultdict(list)

    for query_id, gold_data in gold_annotations.items():
        max_em_score = 0.0
        max_f1_score = 0.0

        gold_strings = gold_data.get("answers_spans", {}).get("spans", [])
        supercat = gold_data.get("super_category", "unknown")

        if not isinstance(gold_strings, list):
            continue

        predicted_full_string = predicted_answers.get(query_id)
        if predicted_full_string is not None:
            extracted_prediction = predicted_full_string.split('\n', 1)[0].strip()
            for gold_string in gold_strings:
                if gold_string.strip():
                    try:
                        em_score, f1 = f1_score(extracted_prediction, gold_string)
                        max_em_score = max(max_em_score, em_score)
                        max_f1_score = max(max_f1_score, f1)
                    except:
                        continue

        instance_exact_match.append(max_em_score)
        instance_f1.append(max_f1_score)
        supercat_scores[supercat].append(max_f1_score)

        if max_f1_score == 1.0:
            correct_ids_by_category["all"].append(query_id)
            correct_ids_by_category[supercat].append(query_id)

    global_em = np.mean(instance_exact_match) if instance_exact_match else 0.0
    global_f1 = np.mean(instance_f1) if instance_f1 else 0.0

    per_super_category = {
        cat: {
            "average_f1": np.mean(scores),
            "average_f1_percent": round(np.mean(scores) * 100, 2),
            "total_evaluated": len(scores)
        }
        for cat, scores in supercat_scores.items()
    }

    return {
        "exact_match": global_em,
        "average_f1": global_f1,
        "average_f1_percent": round(global_f1 * 100, 2),
        "total_evaluated": len(instance_f1),
        "per_super_category": per_super_category,
        "correct_ids_by_category": correct_ids_by_category
    }

# Main model evaluation
def evaluate_models(model_ids):
    results = {}
    gold_path = "eval/golden_ans/gold_answers_by_research_id.json"

    for mid in model_ids:
        model_name = model_id_map[mid]
        pred_path = f"eval/predictions/{model_name}/drop/predictions.json"
        evaluation = evaluate_f1(pred_path, gold_path)

        # Save results JSON
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"drop_{model_name}_{now}.json"
        result_dir = f"eval/results/{model_name}/drop"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, result_filename), "w") as f:
            json.dump({k: v for k, v in evaluation.items() if k != "correct_ids_by_category"}, f, indent=2)

        # Save correct IDs JSON
        correct_dir = f"eval/correct_ans_ids/{model_name}/drop"
        os.makedirs(correct_dir, exist_ok=True)
        correct_filename = f"drop_{model_name}_correct_ids_{now}.json"
        with open(os.path.join(correct_dir, correct_filename), "w") as f:
            json.dump(evaluation["correct_ids_by_category"], f, indent=2)

        results[model_name] = evaluation
    return results

# Run evaluation
evaluate_models(list(model_id_map.keys()))
print("DROP F1 evaluation complete.")
# Note: The evaluation results are saved in the specified directories.