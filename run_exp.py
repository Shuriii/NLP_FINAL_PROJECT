import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import argparse
import os
import json

def duplicate_layers(model, duplication_instructions):
    """Duplicate specific layers in the model according to the instructions."""
    if not duplication_instructions:
        return model

    print("Duplicating layers:", duplication_instructions)

    if hasattr(model.model, 'layers'):  # LLaMA
        layers = model.model.layers
    elif hasattr(model.model, 'transformer'):  # Gemma
        layers = model.model.transformer.h
    else:
        raise ValueError("Model structure not recognized.")

    new_layers = []
    for idx, layer in enumerate(layers):
        new_layers.append(layer)
        for layer_idx, num_dups in duplication_instructions:
            if idx == layer_idx:
                for _ in range(num_dups):
                    new_layers.append(layer)

    if hasattr(model.model, 'layers'):
        model.model.layers = torch.nn.ModuleList(new_layers)
    else:
        model.model.transformer.h = torch.nn.ModuleList(new_layers)

    return model

def load_model(model_name, duplication_instructions):
    is_large = any(x in model_name.lower() for x in ['70b', '27b'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    quantization_config = None

    if is_large:
        print(f"Loading large model {model_name} with 4-bit precision (4bfp).")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,               # <<<<<<<<<<<<<<<< 4bit
            bnb_4bit_quant_type="nf4",        # (better quantization type for LLaMA models)
            bnb_4bit_compute_dtype=torch.bfloat16,  # you can also use float16 if bfloat16 isn't supported
        )
    
    else:
        print(f"Loading smaller model {model_name} with fp16.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map= "auto",
        torch_dtype=torch.float16 if not is_large else None,
        quantization_config=quantization_config,
        use_auth_token=True
    )

    model.eval()

    if duplication_instructions:
        model = duplicate_layers(model, duplication_instructions)

    return tokenizer, model, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Model name to load from HuggingFace.")
    parser.add_argument('--duplications', type=str, default=None,
                        help="Optional duplications: list of tuples as string, e.g. '[(5,2),(10,1)]' meaning duplicate layer 5 twice, layer 10 once.")

    args = parser.parse_args()

    model_name = args.model_name
    duplication_instructions = eval(args.duplications) if args.duplications else None

    seed = 43
    torch.manual_seed(seed)

    os.makedirs('outputs', exist_ok=True)

    tokenizer, model, device = load_model(model_name, duplication_instructions)

    datasets_to_run = ['musique']

    for dataset_name in datasets_to_run:
        print(f"Running dataset {dataset_name}...")
        dataset = load_dataset("sharonsaban/"+dataset_name)

        generations = {}

        start_time = time.time()

        for idx, example in enumerate(dataset['train']):
            print("======================================================================")
            print(f"Processing example {idx + 1}/{len(dataset)}...")
            print(f"Example ID: {example['id']}")
            print(f"Input: {example['input']}")
            input_text = example["input"]  # Always take 'input' field
            example_id = example["id"]      # Always take 'id' field

            input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)
            
            with torch.no_grad():
                outputs = model.generate(input,
                                        max_new_tokens=20,
                                        do_sample=False,
                                        top_p=0,
                                        top_k=0,
                                        temperature=1)
                
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("----------------------------------------------------------------")
            print(f"output: {decoded}")
            print("----------------------------------------------------------------")

            generations[example_id] = decoded

        total_time = time.time() - start_time
        avg_time_per_example = total_time / len(dataset['train'])

        print(f"Finished {dataset_name}:")
        print(f"    Total time: {total_time:.2f} seconds")
        print(f"    Avg time per example: {avg_time_per_example:.4f} seconds")

        # Save outputs + timing
        # if the model is with layer duplication, save the model name with _duplication
        if duplication_instructions:
            model_name = f"{model_name}_duplication"
        output_path = os.path.join("outputs", f"{model_name}_{dataset_name}.json")
        # Create the output data structure
        output_data = {
            "generations": generations,
            "timing": {
                "total_time_seconds": total_time,
                "avg_time_per_example_seconds": avg_time_per_example
            }
        }
        # Save the generations and timing information to a JSON file
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the output data to a JSON file
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved generations and timing to {output_path}")

if __name__ == "__main__":
    main()
