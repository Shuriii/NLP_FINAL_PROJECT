import os
import sys
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_predictions(model, tokenizer, dataset, dataset_name, output_dir, device, max_examples=-1):
    generations = {}

    for i, example in enumerate(dataset):
        if 0 < max_examples == i:
            break

        input_text = example["input"]
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=False)

        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generations[example["id"]] = predicted_text

    output_path = os.path.join(output_dir, f"preds_{dataset_name}.json")
    with open(output_path, 'w') as f_out:
        import json
        json.dump(generations, f_out, indent=4)

    print(f"Saved {len(generations)} predictions to {output_path}")

def main(model_name, output_dir="generations", max_examples=-1, seed=43):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    large_models = ["meta-llama/Meta-Llama-3-70B", "google/gemma-27b"]
    is_large_model = model_name in large_models

    if is_large_model:
        print(f"Loading {model_name} with 8-bit precision")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    else:
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    print("loaded model!") 
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs(output_dir, exist_ok=True)
    # load musique datasets from Huggingface tau/zero_scrolls
    datasets = ["musique"]

    for dataset_name in datasets:
        print(f"Loading {dataset_name} from Huggingface")
        dataset = load_dataset("tau/zero_scrolls", dataset_name, split="train")
        print(f"{dataset_name} loaded!")
        generate_predictions(model, tokenizer, dataset, dataset_name, output_dir, device, max_examples)

if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)
