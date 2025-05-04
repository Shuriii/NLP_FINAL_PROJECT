import json
import os
import importlib.util

# ===> Fill these in:
input_py_file = "eval/scripts/model_registry.py"      # e.g., "scripts/model_registry.py"
output_folder = "eval/scripts/"                # e.g., "data/shared"

# ===> Name of the output JSON
output_filename = "model_registry.json"
output_path = os.path.join(output_folder, output_filename)

# Load the .py file as a module
spec = importlib.util.spec_from_file_location("model_module", input_py_file)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# Extract the three maps
model_id_map = model_module.model_id_map
model_config_map = model_module.model_config_map
configuration_attribute_map = model_module.configuration_attribute_map

# Convert model_id_map keys to strings (JSON requires string keys)
model_id_map_str_keys = {str(k): v for k, v in model_id_map.items()}

# Combine and save
final_dict = {
    "model_id_map": model_id_map_str_keys,
    "model_config_map": model_config_map,
    "configuration_attribute_map": configuration_attribute_map
}

os.makedirs(output_folder, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(final_dict, f, indent=2)

print(f"✅ Maps exported to: {output_path}")
