import string
from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple, Union, Optional
from collections import defaultdict
from model_registry import model_id_map
import numpy as np
from scipy.optimize import linear_sum_assignment


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
    """Lower text and remove punctuation, articles and extra whitespace."""

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
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
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
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1

def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False

def answer_json_to_strings(answer):
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        date_str = "{0} {1} {2}".format(answer["date"]["day"], answer["date"]["month"], answer["date"]["year"])
        parts = [part for part in [answer["date"].get("day"), answer["date"].get("month"), answer["date"].get("year")] if part]
        if parts:
             return tuple([" ".join(parts)]), "date"
        else:
             return tuple([""]), "date"

    else: 
        if isinstance(answer, dict):
            
             potential_answer = " ".join(str(v) for v in answer.values() if isinstance(v, (str, int, float)))
             if potential_answer:
                 return tuple([potential_answer]), "unknown" 
             else: 
                  return tuple([""]), "empty"
        elif isinstance(answer, (str, int, float)):
             return tuple([str(answer)]), "span" 
        else:
            print(f"Warning: Unexpected answer format encountered: {answer}")
            return tuple([""]), "error" 


def normalize(text):
    """Normalize the text using the DROP evaluation logic."""
    return _normalize_answer(text)

def f1_score(prediction, ground_truth):
    """
    Computes Exact Match and F1 score between a prediction and a ground truth answer.
    Uses the DROP evaluation logic. Returns (exact_match, f1).
    """
    predicted_bags = _answer_to_bags(prediction)
    gold_bags = _answer_to_bags(ground_truth)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1

def evaluate_f1(preds_path, gold_path):
    """Evaluates predictions against gold answers using F1 score.

    Args:
        preds_path: Path to the prediction JSON file. {query_id: prediction_string_with_context, ...}
        gold_path: Path to the gold answers JSON file. {query_id: {"answers_spans": {"spans": [...], "types": [...]}}}.

    Returns:
        A dictionary containing 'exact_match' and 'f1' scores.
    """
    try:
        predicted_answers = json.load(open(preds_path, encoding='utf-8'))
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {preds_path}")
        return {"exact_match": 0.0, "f1": 0.0}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from prediction file at {preds_path}")
        return {"exact_match": 0.0, "f1": 0.0}

    try:
        gold_annotations = json.load(open(gold_path, encoding='utf-8'))
    except FileNotFoundError:
        print(f"Error: Gold file not found at {gold_path}")
        return {"exact_match": 0.0, "f1": 0.0}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from gold file at {gold_path}")
        return {"exact_match": 0.0, "f1": 0.0}

    instance_exact_match = []
    instance_f1 = []

    for query_id, gold_data in gold_annotations.items():
        max_em_score = 0.0
        max_f1_score = 0.0

        gold_strings = []
        if isinstance(gold_data, dict) and 'answers_spans' in gold_data and \
           isinstance(gold_data['answers_spans'], dict) and 'spans' in gold_data['answers_spans'] and \
           isinstance(gold_data['answers_spans']['spans'], list):
            gold_strings = gold_data['answers_spans']['spans']
        else:
            print(f"Warning: Invalid format for gold data for query_id {query_id}. Skipping.")
            instance_exact_match.append(0.0)
            instance_f1.append(0.0)
            continue

        predicted_full_string = predicted_answers.get(query_id)

        if predicted_full_string is not None and gold_strings:
            parts = predicted_full_string.split('\n', 1)
            extracted_prediction = parts[0].strip() if parts else ""

            for gold_string in gold_strings:
                if gold_string.strip() == "":
                    continue
                try:
                    em_score, f1 = f1_score(extracted_prediction, gold_string)
                    max_em_score = max(max_em_score, em_score)
                    max_f1_score = max(max_f1_score, f1)
                except Exception as e:
                     print(f"Warning: Error calculating F1 for query_id {query_id}. Pred: '{extracted_prediction}', Gold: '{gold_string}'. Error: {e}")
                     continue

        elif not gold_strings:
            print(f"Warning: No gold answer spans found for query_id {query_id}")
            pass
        elif predicted_full_string is None:
            print(f"Warning: Missing prediction for question: {query_id}")

        instance_exact_match.append(max_em_score)
        instance_f1.append(max_f1_score)

    global_em = np.mean(instance_exact_match) if instance_exact_match else 0.0
    global_f1 = np.mean(instance_f1) if instance_f1 else 0.0

    print(f"Evaluated {len(instance_f1)} instances.")
    print(f"Exact-match accuracy {global_em * 100:.2f}")
    print(f"F1 score {global_f1 * 100:.2f}")

    return {"exact_match": global_em, "f1": global_f1}

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



evaluate_models([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
# # evaluate_models(list(model_id_map.keys()))
# print("DROP F1 evaluation complete.")