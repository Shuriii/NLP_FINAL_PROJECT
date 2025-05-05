import os

# ===> Replace this with your target path
root_path = "eval/results/"

# Iterate through each model folder inside the root path
for model_folder in os.listdir(root_path):
    model_path = os.path.join(root_path, model_folder)
    if not os.path.isdir(model_path):
        continue  # Skip files, only process directories

    # List of dataset subfolders to process
    for dataset in ["drop", "mmlu", "musique"]:
        dataset_path = os.path.join(model_path, dataset)
        if not os.path.isdir(dataset_path):
            continue  # Skip if dataset folder does not exist

        # Delete all files inside the dataset folder
        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

print("All files inside drop, mmlu, and musique folders have been deleted.")
