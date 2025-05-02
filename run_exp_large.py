import time
import torch
import os
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import vllm

# Initialize FastAPI server
app = FastAPI()

# Global variables
is_large = False
num_layers_original = 0
model_name = ""

# Model loading function using vLLM for multi-GPU inference
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model {model_name} using vLLM with multi-GPU support.")
    model = vllm.ModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically distribute the model across GPUs
        load_in_8bit=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model loaded successfully")
    return model, tokenizer, device

# Request model inference (for FastAPI)
class InferenceRequest(BaseModel):
    input_text: str

@app.post("/generate/")
async def generate_text(request: InferenceRequest):
    input_text = request.input_text
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).input_ids.to(model.device)
    
    with torch.inference_mode():
        start_time = time.time()
        generated_ids = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        end_time = time.time()

    run_time = end_time - start_time
    print(f"Inference run time: {run_time:.2f} seconds")
    return {"output_text": output_text, "run_time": run_time}

# Inference on all datasets
def process_datasets(model, tokenizer, device, datasets_to_run):
    predictions = {}
    run_times = {}
    model_name_to_save = model_name.split("/")[-1]  # Extract model name for saving results
    for dataset_name in datasets_to_run:
        print(f"Loading dataset: {dataset_name}")
        
        # Assuming your datasets are JSON files stored under `data_sets/{dataset_name}/input_prompt_samples/`
        dataset_path = f"data_sets/{dataset_name}/input_prompt_samples/"
        dataset = []
        
        if os.path.exists(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith(".json"):
                    with open(os.path.join(dataset_path, file), 'r') as f:
                        dataset += json.load(f)
        
        print(f"Loaded {len(dataset)} examples from {dataset_name}")

        os.makedirs(f"results/{model_name_to_save}/{dataset_name}/attentions", exist_ok=True)
        os.makedirs(f"results/{model_name_to_save}/{dataset_name}/logits", exist_ok=True)
        os.makedirs(f"results/{model_name_to_save}/{dataset_name}/hidden_states", exist_ok=True)
        
        # Process each example in the dataset
        for idx, example in enumerate(dataset):
            print(f"Processing example {idx + 1}/{len(dataset)}...")

            input_text = example["input"]  # Always take 'input' field
            example_id = example["research_id"]  # Always take 'id' field

            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)

            with torch.inference_mode():
                start_time = time.time()
                generated_ids = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=False)
                output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                end_time = time.time()

                run_time = end_time - start_time
                print(f"Inference run time: {run_time:.2f} seconds")
                
                # Save the results to JSON files
                predictions[example_id] = output_text
                run_times[example_id] = run_time

                # Optionally, save the logits, attentions, and hidden states if needed
                # You can enable saving as needed
                # with open(f"results/{dataset_name}/logits/{example_id}.json", "w") as f:
                #     json.dump(logits.tolist(), f, indent=2)
                # with open(f"results/{dataset_name}/attentions/{example_id}.json", "w") as f:
                #     json.dump(attentions, f, indent=2)
                # with open(f"results/{dataset_name}/hidden_states/{example_id}.json", "w") as f:
                #     json.dump(hidden_states, f, indent=2)

        # Save all the results for this dataset
        with open(f"results/{dataset_name}/predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)
        with open(f"results/{dataset_name}/run_times.json", "w") as f:
            json.dump(run_times, f, indent=2)
        print(f"Finished processing {dataset_name}")

# Load model for running inference
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="Model name to load from HuggingFace.")
    args = parser.parse_args()

    model_name = args.model_name

    print(f"Starting inference for model: {model_name}")

    # Load model and tokenizer
    model, tokenizer, device = load_model(model_name)
    # verify you use multi-gpu
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
    else:
        print("Using a single GPU for inference.")
    # Datasets to run inference on
    datasets_to_run = ['musique', 'mmlu', 'drop']

    # Process the datasets
    process_datasets(model, tokenizer, device, datasets_to_run)

if __name__ == "__main__":
    main()
