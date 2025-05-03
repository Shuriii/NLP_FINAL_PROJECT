import time
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import dispatch_model, infer_auto_device_map
import argparse
import os
import json

is_large = False
num_layers_original = 0

def duplicate_layers(model, duplication_instructions, device):
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
    # Update the model config to reflect the new number of layers
    model.config.num_hidden_layers = len(new_layers)
    # move the model to the device
    
    return model

def load_model(model_name, duplication_instructions):
    is_large = any(x in model_name.lower() for x in ['70b', '27b'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    quantization_config = None

    if is_large:
        print(f"Loading large model {model_name} with 8-bit precision (8bfp).")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print(f"Using quantization config: {quantization_config}")
    
    else:
        print(f"Loading smaller model {model_name} with fp16.")

    print(f"loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=True, force_download=True)    
    tokenizer.pad_token = tokenizer.eos_token
    print(f"loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map= None,
        torch_dtype=torch.float16 if not is_large else None,
        quantization_config=quantization_config,
        use_auth_token=True,
        force_download=True
    )
    print(f"loaded model")
    num_layers_original = len(model.model.layers) if hasattr(model.model, 'layers') else len(model.model.transformer.h)
    print(f"Original number of layers: {num_layers_original}")
    model.eval()

    if duplication_instructions:
        model = duplicate_layers(model, duplication_instructions, device)

    if "llama" in model_name.lower():
        no_split_modules = ["LlamaDecoderLayer"]
    elif "gemma" in model_name.lower():
        no_split_modules = ["GemmaDecoderLayer"]
    else:
        raise ValueError("Unknown model type for no_split_module_classes.")

    device_map = infer_auto_device_map(
        model,
        max_memory = {i: "23GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=no_split_modules
    )

    model = dispatch_model(model, device_map=device_map)
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
  

    datasets_to_run = ['musique','mmlu','drop']

    if duplication_instructions:
            model_name_to_save = f"{model_name.split('/')[1]}_duplication_{duplication_instructions}"
    else:
            model_name_to_save = model_name.split('/')[1]
    # drop spaces in model_name_to_save
    model_name_to_save = model_name_to_save.replace(" ", "")
    
    for dataset_name in datasets_to_run:
        # load the datasets manually
        # search for the json files in the dataset folder
        # open the dataset folder and load the json files dont use load disk
        print("loading dataset")
        dataset_path = f"data_sets/{dataset_name}/input_prompt_samples/"
        if os.path.exists(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith(".json"):
                    with open(os.path.join(dataset_path, file), 'r') as f:
                        dataset = json.load(f)
        print(f"loaded dataset {dataset_name}")

        print(f"dataset size: {len(dataset)}")
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

        for idx, example in enumerate(dataset):
            print("#######################################################################")
            print(f"Processing example {idx + 1}/{len(dataset)}...")
            print(f"Example ID: {example['research_id']}")
            print(f"Input: {example['input']}")
            input_text = example["input"]  # Always take 'input' field
            example_id = example["research_id"]      # Always take 'id' field

            input = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
 
            with torch.inference_mode():
                start_time = time.time()
                top_p = 0
                top_k = 0
                temperature = 1

                # get the logits, attentions, and hidden states
                input_ids = input["input_ids"].to(device)
        #        outputs = model(input_ids=input_ids,
        #                    return_dict=True,
        #                        output_attentions=True,
        #                        output_hidden_states=True)
        #        
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature)
                
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(input_text):]
                print("got an output")
                end_time = time.time()
                run_time = end_time - start_time
                print(f"Run time: {run_time:.2f} seconds")
                
        #        logits = outputs.logits.detach().cpu().numpy()
        #        attentions = outputs.attentions
        #        hidden_states = outputs.hidden_states

                # Save the logits, attentions, and hidden states to json files
        #        with open(f"results/{model_name_to_save}/{dataset_name}/logits/{example_id}.json", "w") as f:
        #            logits_data = logits.tolist()  # Convert numpy array to list for JSON serialization
        #            json.dump(logits_data, f, indent=2)
        #        with open(f"results/{model_name_to_save}/{dataset_name}/attentions/{example_id}.json", "w") as f:
        #            attentions_data = [attn.tolist() for attn in attentions]  # Convert tensors to lists
        #            json.dump(attentions_data, f, indent=2)
        #        with open(f"results/{model_name_to_save}/{dataset_name}/hidden_states/{example_id}.json", "w") as f:
        #            hidden_states_data = [hidden_state.tolist() for hidden_state in hidden_states]
        #            json.dump(hidden_states_data, f, indent=2)

                print("saved the logits, attentions, and hidden states to json files")
            # get the output text from 
           
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
