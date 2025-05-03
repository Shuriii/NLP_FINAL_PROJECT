import string
from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

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



def _remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text):
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def _remove_punc(text):
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def _lower(text):
    return text.lower()

def _tokenize(text):
    return re.split(" |-", text)

def _normalize_answer(text):
    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text):
    if _is_number(text):
        return str(float(text))
    else:
        return text

def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
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
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return f1

def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set(word for word in gold_bag if _is_number(word))
    predicted_numbers = set(word for word in predicted_bag if _is_number(word))
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))

def normalize(text):
    return _normalize_answer(text)

def f1_score(prediction, ground_truth):
    predicted_bags = _answer_to_bags(prediction)
    gold_bags = _answer_to_bags(ground_truth)
    exact_match = 1.0 if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]) else 0.0
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = round(np.mean(f1_per_bag), 2)
    return exact_match, f1

def evaluate_f1(preds_path, gold_path):
    try:
        predicted_answers = json.load(open(preds_path, encoding='utf-8'))
        gold_annotations = json.load(open(gold_path, encoding='utf-8'))
    except Exception as e:
        print(f"Error loading files: {e}")
        return {"exact_match": 0.0, "f1": 0.0, "total": 0, "per_super_category": {}}

    instance_exact_match = []
    instance_f1 = []
    supercat_scores = defaultdict(list)

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

    global_em = np.mean(instance_exact_match) if instance_exact_match else 0.0
    global_f1 = np.mean(instance_f1) if instance_f1 else 0.0

    print(f"Evaluated {len(instance_f1)} instances.")
    print(f"Exact-match accuracy {global_em * 100:.2f}")
    print(f"F1 score {global_f1 * 100:.2f}")

    per_super_category = {}
    for cat, scores in supercat_scores.items():
        avg = np.mean(scores) if scores else 0.0
        per_super_category[cat] = {
            "average_f1": avg,
            "average_f1_percent": round(avg * 100, 2),
            "total_evaluated": len(scores)
        }

    return {
        "exact_match": global_em,
        "average_f1": global_f1,
        "average_f1_percent": round(global_f1 * 100, 2),
        "total_evaluated": len(instance_f1),
        "per_super_category": per_super_category
    }

def evaluate_models(model_ids):
    results = {}
    gold_path = "eval/golden_ans/gold_answers_by_research_id.json"
    for mid in model_ids:
        model_name = model_id_map[mid]
        pred_path = f"eval/predictions/{model_name}/drop/predictions.json"
        result = evaluate_f1(pred_path, gold_path)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"drop{model_name}_{now}.json"
        result_dir = f"eval/results/{model_name}/drop"
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, result_filename), "w") as f:
            json.dump(result, f, indent=2)

        results[model_name] = result
    return results

#evaluate_models([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
print("DROP F1 evaluation complete.")
evaluate_models(list(model_id_map.keys()))