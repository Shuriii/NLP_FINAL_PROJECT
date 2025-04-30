import time
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import argparse
import os
import json

is_large = False
num_layers_original = 0

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

    print(f"Original number of layers: {num_layers_original}")
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
        attn_implementation="eager",
        torch_dtype=torch.float16 if not is_large else None,
        quantization_config=quantization_config,
        use_auth_token=True
    )
    num_layers_original = len(model.model.layers) if hasattr(model.model, 'layers') else len(model.model.transformer.h)
    print(f"Original number of layers: {num_layers_original}")
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
    # print the model config
    print("model config:")
    print(model.config)
  

    datasets_to_run = ['musique', 'mmlu']

    if duplication_instructions:
            model_name_to_save = f"{model_name}_duplication_{duplication_instructions}"
    else:
            model_name_to_save = model_name
        
    for dataset_name in datasets_to_run:
        try:
            print(f"loading dataset {dataset_name} from disk...")
            # Load the dataset from disk if it exists
            dataset = load_from_disk(f"datasets/{dataset_name}/")
            print(f"loaded dataset {dataset_name} from disk")
        except:
            print(f"loading dataset {dataset_name} from huggingface...")
            dataset = load_dataset("sharonsaban/"+dataset_name, cache_dir="./hf_cache")
            dataset.save_to_disk(f"datasets/{dataset_name}/")
            print(f"loaded dataset {dataset_name} from huggingface")
 
        print(f"dataset size: {len(dataset['train'])}")
        predictions = {}
        run_times = {}
        if dataset_name == 'mmlu':
            max_new_tokens = 1
        else:
            max_new_tokens = 128
     
        # save the logits, attentions, hidden_states, and run_times to the model_name/dataset_name folder json file for each catagory
        os.makedirs(f"results/{model_name_to_save}/{dataset_name}/logits", exist_ok=True)
        os.makedirs(f"results/{model_name_to_save}/{dataset_name}/attentions", exist_ok=True)
        os.makedirs(f"results/{model_name_to_save}/{dataset_name}/hidden_states", exist_ok=True)

        for idx, example in enumerate(dataset['train']):
            print("#######################################################################")
            print(f"Processing example {idx + 1}/{len(dataset)}...")
            print(f"Example ID: {example['id']}")
            print(f"Input: {example['input']}")
            input_text = example["input"]  # Always take 'input' field
            example_id = example["id"]      # Always take 'id' field

            input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)
 
            with torch.no_grad():
                start_time = time.time()
                top_p = 0
                top_k = 0
                temperature = 1

                output = model(input,
                                output_attentions=True,
                                output_hidden_states=True,
                                return_dict=True,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                top_p=top_p,
                                top_k=top_k,
                                temperature=temperature)
                print("got an output")
                end_time = time.time()
                run_time = end_time - start_time
                
                logits = output.logits.detach().cpu().numpy()
                attentions = output.attentions
                hidden_states = output.hidden_states

                # Save the logits, attentions, and hidden states to json files
                with open(f"results/{model_name_to_save}/{dataset_name}/logits/{example_id}.json", "w") as f:
                    logits_data = logits.tolist()  # Convert numpy array to list for JSON serialization
                    json.dump(logits_data, f, indent=2)
                with open(f"results/{model_name_to_save}/{dataset_name}/attentions/{example_id}.json", "w") as f:
                    attentions_data = [attn.tolist() for attn in attentions]  # Convert tensors to lists
                    json.dump(attentions_data, f, indent=2)
                with open(f"results/{model_name_to_save}/{dataset_name}/hidden_states/{example_id}.json", "w") as f:
                    hidden_states_data = [hidden_state.tolist() for hidden_state in hidden_states]
                    json.dump(hidden_states_data, f, indent=2)

                print("saved the logits, attentions, and hidden states to json files")

            output_text = tokenizer.decode(output.logits[0, -1, :].argmax(-1).item(), skip_special_tokens=True)
            print("----------------------------------------------------------------")
            print(f"output: {output_text}")
            print("----------------------------------------------------------------")

            predictions[example_id] = output_text
            run_times[example_id] = run_time

        with open(f"results/{model_name_to_save}/{dataset_name}/predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)
        with open(f"results/{model_name_to_save}/{dataset_name}/run_times.json", "w") as f:
            json.dump(run_times, f, indent=2)
        # save all the hyper parameters to the model_name/dataset_name folder json file
        with open(f"results/{model_name_to_save}/{dataset_name}/run_config.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "duplication_instructions": duplication_instructions,
                "dataset_name": dataset_name,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "precision:": "fp16" if not is_large else "4bfp",
                "device": str(device),
                "num_layers": len(model.model.layers) if hasattr(model.model, 'layers') else len(model.model.transformer.h),
                "num_layers_original": num_layers_original,
                "model_config": model.config.to_dict(),
                "tokenizer_config": tokenizer.get_vocab()
            }, f, indent=2)

        print(f"Finished {dataset_name}:")


    
if __name__ == "__main__":
    main()
