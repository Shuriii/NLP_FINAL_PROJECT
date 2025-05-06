import os
import json

# === Paths ===
RAW_INPUT_PATH = "data_sets/drop/drop_raw_data_samples/drop_500_samples.json"
FIVESHOT_DIR = "data_sets/drop/drop_5shot_samples"
OUTPUT_DIR = "data_sets/drop/drop_input_prompt_ samples"
OUTPUT_FILENAME = "drop_reformatted_samples.json"

# === Fixed Intro ===
INTRO = (
    "You are a reading comprehension assistant.\n"
    "For each example, you are given a passage and a question.\n"
    "Base your answer solely on the information in the passage.\n"
    "Answer as concisely as possible, using only a few words or a number.\n\n"
)

# === Helper Functions ===
def get_super_category(section_id):
    if section_id.startswith("history_"):
        return "history"
    elif section_id.startswith("nfl_"):
        return "nfl"
    return "unknown"

def format_example(example):
    passage = example["passage"].strip()
    question = example["question"].strip()
    answer = example["answers_spans"]["spans"][0].strip() if example["answers_spans"]["spans"] else ""
    return f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}\n"

def format_input_prompt(five_examples, target_sample):
    formatted = INTRO
    for ex in five_examples:
        formatted += format_example(ex) + "\n"
    formatted += f"Passage: {target_sample['passage'].strip()}\n"
    formatted += f"Question: {target_sample['question'].strip()}\n"
    formatted += "Answer:"
    return formatted

# === Main Logic ===
def build_drop_samples():
    with open(RAW_INPUT_PATH, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    formatted_samples = []

    for i, sample in enumerate(raw_samples):
        section_id = sample["section_id"]
        query_id = sample["query_id"]
        category = get_super_category(section_id)

        # Load five-shot examples
        five_shot_path = os.path.join(FIVESHOT_DIR, f"{category}_5shot.json")
        with open(five_shot_path, "r", encoding="utf-8") as f:
            five_shot_examples = json.load(f)

        input_prompt = format_input_prompt(five_shot_examples, sample)

        formatted_sample = {
            "research_id": 1001 + i,
            "section_id": section_id,
            "query_id": query_id,
            "super_category": category,
            "input": input_prompt,
            "output": {
                "answers_spans": sample.get("answers_spans", {})
            }
        }

        formatted_samples.append(formatted_sample)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_samples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(formatted_samples)} formatted samples to: {output_path}")

# === Run ===
build_drop_samples()
