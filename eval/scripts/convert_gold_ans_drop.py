import json
import os

# ===> Fill these in before running
input_path = "data_sets/drop/drop_input_prompt_ samples/drop_reformatted_samples.json"   # ← Your file like in image 1
output_folder = "eval/golden_ans"           # ← The folder to save output in
output_filename = "drop_gold_ans_try1.json"
output_path = os.path.join(output_folder, output_filename)

# Load your data
with open(input_path, "r") as f:
    custom_data = json.load(f)

# Convert to DROP-style format
drop_format_data = {}
for item in custom_data:
    section_id = item["section_id"]
    if section_id not in drop_format_data:
        drop_format_data[section_id] = {
            "passage": "",
            "qa_pairs": []
        }

    # Extract passage
    input_text = item["input"]
    passage_start = input_text.find("Passage:")
    question_start = input_text.find("\nQuestion:")
    passage = input_text[passage_start + len("Passage:"):question_start].strip() if passage_start != -1 else ""
    drop_format_data[section_id]["passage"] = passage

    # Build QA entry
    question = input_text[question_start + len("\nQuestion:"):].split("\nAnswer:")[0].strip()
    spans = item.get("output", {}).get("answers_spans", {}).get("spans", [])
    qa_entry = {
        "question": question,
        "answer": {
            "number": "",  # leave empty
            "date": {"day": "", "month": "", "year": ""},
            "spans": spans
        },
        "query_id": item.get("query_id", "")
    }
    drop_format_data[section_id]["qa_pairs"].append(qa_entry)

# Save output
os.makedirs(output_folder, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(drop_format_data, f, indent=2)

print(f"Saved DROP-style file to: {output_path}")
