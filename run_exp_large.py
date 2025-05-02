import time
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import dispatch_model, infer_auto_device_map
import argparse
import os
import json
from vllm import LLM, SamplingParams

is_large = False
num_layers_original = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = LLM(
        model_name,
        tensor_parallel_size=4,
        torch_dtype=torch.float16,
        quantization="fp8",
        use_auth_token=True

    )
    # print the model config
    print("model config:")
    print(model.config)
  

    datasets_to_run = ['musique','mmlu','drop']

    if duplication_instructions:
            model_name_to_save = f"{model_name.split('/')[1]}_duplication_{duplication_instructions}"
    else:
            model_name_to_save = model_name.split('/')[1]
    top_p = 0
    top_k = 0
    temperature = 1.0

    SamplingParams = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k)
    
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
            start_time = time.time()
            output_text = model.generate(input_text, sampling_params=SamplingParams)
            end_time = time.time()
            run_time = end_time - start_time
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
                "model_config": model.config.to_dict()
            }, f, indent=2)

        print(f"Finished {dataset_name}:")


    
if __name__ == "__main__":
    main()
