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
    print(f"Duplicated layers: {len(new_layers)} total layers (original: {len(layers)})")
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
        output_attentions=True, 
        output_hidden_states=True, 
        return_dict=True,
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

    os.makedirs('results', exist_ok=True)

    tokenizer, model, device = load_model(model_name, duplication_instructions)

    datasets_to_run = ['musique', 'mmlu']

    for dataset_name in datasets_to_run:
        print(f"Running dataset {dataset_name}...")
        dataset = load_dataset("sharonsaban/"+dataset_name)

        predictions = {}
        run_times = {}

        if duplication_instructions:
            model_name = f"{model_name}_duplication"
     
        # save the logits, attentions, hidden_states, and run_times to the model_name/dataset_name folder json file for each catagory
        os.makedirs(os.path.dirname(f"results/{model_name}"), exist_ok=True)
        os.makedirs(os.path.dirname(f"results/{model_name}/{dataset_name}"), exist_ok=True) 
        os.makedirs(os.path.dirname(f"results/{model_name}/{dataset_name}/logits"), exist_ok=True)
        os.makedirs(os.path.dirname(f"results/{model_name}/{dataset_name}/attentions"), exist_ok=True)
        os.makedirs(os.path.dirname(f"results/{model_name}/{dataset_name}/hidden_states"), exist_ok=True)
     

        for idx, example in enumerate(dataset['train']):
            print("#######################################################################")
            print(f"Processing example {idx + 1}/{len(dataset)}...")
            print(f"Example ID: {example['id']}")
            print(f"Input: {example['input']}")
            input_text = example["input"]  # Always take 'input' field
            example_id = example["id"]      # Always take 'id' field

            input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)
            if dataset_name == 'mmlu':
                max_new_tokens = 1
            else:
                max_new_tokens = 128
            with torch.no_grad():
                start_time = time.time()
                top_p = 0
                top_k = 0
                temperature = 1

                output = model.generate(input,
                                         max_new_tokens=max_new_tokens,
                                         do_sample=False,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature)
                end_time = time.time()
                run_time = end_time - start_time
                
                logits = output.logit.detach().cpu().numpy()         # Save these to analyze model confidence later
                attentions = output.attentions                        # Save these to build attention maps later
                hidden_states = output.hidden_states                  # Save these to compare internal layers later

                with open(f"results/{model_name}/{dataset_name}/logits/{example_id}.json", "w") as f:
                    json.dump(logits, f, indent=2)
                with open(f"results/{model_name}/{dataset_name}/attentions/{example_id}.json", "w") as f:
                    json.dump(attentions, f, indent=2)
                with open(f"results/{model_name}/{dataset_name}/hidden_states/{example_id}.json", "w") as f:
                    json.dump(hidden_states, f, indent=2)


            output_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(input_text):]
            print("----------------------------------------------------------------")
            print(f"output: {output_text}")
            print("----------------------------------------------------------------")

            predictions[example_id] = output_text
            run_times[example_id] = run_time

        with open(f"results/{model_name}/{dataset_name}/predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)
        with open(f"results/{model_name}/{dataset_name}/run_times.json", "w") as f:
            json.dump(run_times, f, indent=2)
        # save all the hyper parameters to the model_name/dataset_name folder json file
        with open(f"results/{model_name}/{dataset_name}/run_config.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "duplication_instructions": duplication_instructions,
                "dataset_name": dataset_name,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "precision": "4bit" if is_large else "fp16",
                "device": str(device),
                "num_layers": len(model.model.layers) if hasattr(model.model, 'layers') else len(model.model.transformer.h),
                "num_layers_original": len(model.model.layers) if hasattr(model.model, 'layers') else len(model.model.transformer.h)
            }, f, indent=2)

        print(f"Finished {dataset_name}:")


    
if __name__ == "__main__":
    main()
