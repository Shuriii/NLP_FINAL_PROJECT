import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import os
import json
import ast

def duplicate_layers(model, duplication_plan):
    if hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model.model, "decoder_layers"):
        layers = model.model.decoder_layers
    else:
        raise ValueError("Unknown model structure")

    print(f"Original number of layers: {len(layers)}")

    offset = 0
    for layer_idx, duplication_count in duplication_plan:
        true_idx = layer_idx + offset
        if true_idx >= len(layers):
            raise ValueError(f"Layer {true_idx} out of range!")

        print(f"Duplicating layer {true_idx} {duplication_count} times...")

        layer_to_copy = layers[true_idx]
        for _ in range(duplication_count):
            copied_layer = type(layer_to_copy)(model.config)
            copied_layer.load_state_dict(layer_to_copy.state_dict())
            layers.insert(true_idx + 1, copied_layer)
            offset += 1

    print(f"New number of layers: {len(layers)}")
    return model

def run_inference(model, tokenizer, dataset_name, output_dir, max_examples=50):
    os.makedirs(output_dir, exist_ok=True)
    data = load_dataset(dataset_name, split="test")

    outputs = {}

    for idx, example in enumerate(data):
        if idx >= max_examples:
            break

        prompt = example["input"] if "input" in example else example.get("question", "")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=256)
        
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs[example.get("id", f"example_{idx}")] = output_text

    dataset_clean_name = dataset_name.replace('/', '_')
    output_file = os.path.join(output_dir, f"{dataset_clean_name}_outputs.json")
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=4)

    print(f"Saved {len(outputs)} generations to {output_file}")

def is_big_model(model_name):
    return ("70b" in model_name.lower()) or ("27b" in model_name.lower())

def main(
    model_name,  # Now SINGLE model name
    duplication_plan = None,
    datasets = ["tau/zero_scrolls/musique", "drop", "llmu"],
):
    random.seed(43)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(duplication_plan, str):
        duplication_plan = ast.literal_eval(duplication_plan)

    print(f"\n==== Running model {model_name} ====")

    big_model = is_big_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        use_auth_token=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True if big_model else False,
        torch_dtype=torch.float16 if not big_model else None,
        use_auth_token=True
    )

    if duplication_plan:
        print(f"Applying duplication plan {duplication_plan}...")
        model = duplicate_layers(model, duplication_plan)

    model.eval()

    for dataset_name in datasets:
        print(f"Running inference on {dataset_name}...")
        output_folder = f"outputs/{model_name.replace('/', '_')}"
        run_inference(model, tokenizer, dataset_name, output_dir=output_folder)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
