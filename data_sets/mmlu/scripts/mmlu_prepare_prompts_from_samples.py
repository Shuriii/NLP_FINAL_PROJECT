import os
import json

# === FILL IN THESE PATHS ===
RAW_INPUT_PATH = "data_sets/mmlu/mmlu_raw_data_ samples/mmlu_500_samples.json"
FIVESHOT_DIR = "data_sets/mmlu/mmlu_5shot_samples"
OUTPUT_DIR = "data_sets/mmlu/mmlu_input_prompt_ samples"
OUTPUT_FILENAME = "mmlu_reformatted_samples.json"

# === Super Category Mapping ===
SUPER_CATEGORIES = {
    "STEM": {
        "Abstract Algebra", "Anatomy", "Astronomy", "College Biology", "College Chemistry", "College Computer Science",
        "College Mathematics", "College Physics", "Computer Security", "Conceptual Physics", "Electrical Engineering",
        "Elementary Mathematics", "High School Biology", "High School Chemistry", "High School Computer Science",
        "High School Mathematics", "High School Physics", "Machine Learning", "High School Statistics"
    },
    "Humanities": {
        "Formal Logic", "High School European History", "High School US History", "High School World History",
        "International Law", "Jurisprudence", "Logical Fallacies", "Moral Disputes", "Moral Scenarios",
        "Philosophy", "Prehistory", "Professional Law", "World Religions"
    },
    "Social_Sciences": {
        "Econometrics", "High School Geography", "High School Government and Politics",
        "High School Macroeconomics", "High School Microeconomics", "High School Psychology", "Public Relations",
        "Security Studies", "Sociology", "Human Sexuality", "Professional Psychology", "US Foreign Policy"
    },
    "Other": {
        "Business Ethics", "Clinical Knowledge", "College Medicine", "Global Facts", "Human Aging",
         "Management", "Marketing", "Medical Genetics", "Miscellaneous", "Nutrition",
        "Professional Accounting", "Professional Medicine", "Virology"
    }
}

# === Answer Mapping ===
ANSWER_MAP = {"0": "A", "1": "B", "2": "C", "3": "D"}

# === Helper Functions ===

def format_subject(subject: str) -> str:
    return " ".join(subject.split("_"))

def get_super_category(subject: str) -> str:
    formatted = format_subject(subject).lower()
    for category, subjects in SUPER_CATEGORIES.items():
        normalized_subjects = {s.lower() for s in subjects}
        if formatted in normalized_subjects:
            return category
    return "Unknown"


def format_question_with_choices(q: dict) -> str:
    text = q["question"].strip()
    for idx, choice in enumerate(q["choices"]):
        letter = ["A", "B", "C", "D"][idx]
        text += f"\n{letter}. {choice.strip()}"
    return text

def format_five_shot_prompt(subject: str, five_shot_examples: list) -> str:
    readable_subject = format_subject(subject)
    header = f"The following are multiple choice questions (with answers) about {readable_subject}.\n"
    instruction = "Please answer only with A, B, C, or D representing the correct answer.\n"
    body = ""
    for example in five_shot_examples:
        q_block = format_question_with_choices(example)
        answer = ANSWER_MAP[str(example["answer"]).strip()]
        body += f"{q_block}\nAnswer: {answer}\n\n"
    return header + instruction + "\n" + body

def format_sample_input(subject: str, five_shot_examples: list, test_sample: dict) -> str:
    prompt = format_five_shot_prompt(subject, five_shot_examples)
    test_block = format_question_with_choices(test_sample)
    return prompt + f"{test_block}\nAnswer:"

def convert_answer(raw_answer: str) -> str:
    return ANSWER_MAP[str(raw_answer).strip()]

def build_mmlu_samples():
    with open(RAW_INPUT_PATH, "r") as f:
        raw_samples = json.load(f)

    formatted_samples = []

    for i, sample in enumerate(raw_samples):
        subject = sample["subject"]
        five_shot_path = os.path.join(FIVESHOT_DIR, f"{subject}_5shot.json")

        with open(five_shot_path, "r") as f:
            five_shot_examples = json.load(f)

        input_text = format_sample_input(subject, five_shot_examples, sample)
        correct_answer = convert_answer(sample["answer"])
        super_category = get_super_category(subject)

        formatted_sample = {
            "research_id": i + 501,
            "subject": subject,
            "super_category": super_category,
            "input": input_text,
            "output": correct_answer
        }

        formatted_samples.append(formatted_sample)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        json.dump(formatted_samples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(formatted_samples)} formatted samples to: {output_path}")

# === Run the code ===
build_mmlu_samples()
