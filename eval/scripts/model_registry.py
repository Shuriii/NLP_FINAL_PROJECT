model_id_map = {
    1: "gemma-2-2b",
    2: "gemma-2-2b_duplication_[(0,1),(25,1)]",
    3: "gemma-2-2b_duplication_[(12,1),(13,1)]",
    4: "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]",
    5: "gemma-2-2b_duplication_[(0,3),(22,3)]",
    6: "gemma-2-2b_duplication_[(0,4)]",
    7: "gemma-2-2b_duplication_[(10,5)]",
    8: "gemma-2-2b_duplication_[(11,3)]",
    9: "gemma-2-2b_duplication_[(13,2)]",
    10: "gemma-2-2b_duplication_[(22,4)]",
    11: "gemma-2-9b",
  12: "gemma-2-27b",
    13: "Meta-Llama-3-8B",
    14: "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]",
    15: "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]",
    16: "Meta-Llama-3-8B_duplication_[(0,4),(27,4)]",
    17: "Meta-Llama-3-8B_duplication_[(0,5)]",
    18: "Meta-Llama-3-8B_duplication_[(13,5)]",
    19: "Meta-Llama-3-8B_duplication_[(14,3)]",
    20: "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]",
    21: "Meta-Llama-3-8B_duplication_[(16,2)]",
    22: "Meta-Llama-3-8B_duplication_[(27,5)]",
23: "Meta-Llama-3-70B",
    24: "Meta-Llama-3-8B_duplication_[(26,3),(29,3)]",
    25: "gemma-2-2b_duplication_[(24,2)]",
    26: "gemma-2-2b_duplication_[(23,1),(24,1),(25,1),(26,1),(27,1)]",
    27: "gemma-2-2b_duplication_[(22,3),(25,3)]",
    28: "gemma-2-2b_duplication_[(21,7)]",
    29: "gemma-2-2b_duplication_[(14,2),(16,2),(18,2)]",
    30: "gemma-2-2b_duplication_[(14,2),(14,2),(16,2),(16,2)]",
    31: "gemma-2-2b_duplication_[(13,1),(14,1),(15,1)]",
    32: "gemma-2-2b_duplication_[(12,3),(15,3)]",
    33: "gemma-2-2b_duplication_[(12,1),(13,1),(14,1),(15,1)]",
    34: "Meta-Llama-3-8B_duplication_[(28,2)]",
    35: "Meta-Llama-3-8B_duplication_[(27,1),(28,1),(29,1),(30,1),(31,1)]",
    36: "Meta-Llama-3-8B_duplication_[(25,7)]",
    37: "Meta-Llama-3-8B_duplication_[(16,2),(18,2),(20,2)]",
    38: "Meta-Llama-3-8B_duplication_[(16,2),(16,2),(18,2),(18,2)]",
    39: "Meta-Llama-3-8B_duplication_[(15,1),(16,1),(17,1)]",
    40: "Meta-Llama-3-8B_duplication_[(14,3),(17,3)]",
    41: "Meta-Llama-3-8B_duplication_[(14,1),(15,1),(16,1),(17,1)]"
    

}

# Model mapping (reloaded after kernel reset)


# Model configuration mapping
model_config_map = {
    "gemma-2-2b": "original",
    "gemma-2-2b_duplication_[(0,1),(25,1)]": "G",  #first, last/ in_place / small/ 
    "gemma-2-2b_duplication_[(12,1),(13,1)]": "H", #middle/ in_place / small / true
    "gemma-2-2b_duplication_[(0,1),(12,1),(25,1)]": "I", #first, middle, last/ in_place / small
    "gemma-2-2b_duplication_[(0,3),(22,3)]": "F", # first, last/ blocks / medium
    "gemma-2-2b_duplication_[(0,4)]": "D", # first/ blocks / big
    "gemma-2-2b_duplication_[(10,5)]": "C", # middle/ blocks / big
    "gemma-2-2b_duplication_[(11,3)]": "A", #middle/ blocks / medium
    "gemma-2-2b_duplication_[(13,2)]": "B", #middle/ in_place / medium
    "gemma-2-2b_duplication_[(22,4)]": "E", #last/ blocks / medium
    "gemma-2-9b": "original", 
    "Meta-Llama-3-8B": "original", 
    "Meta-Llama-3-8B_duplication_[(0,1),(15,1),(31,1)]": "I", #first, middle, last/ in_place / small
    "Meta-Llama-3-8B_duplication_[(0,1),(31,1)]": "G", #first, last/ in_place / small
    "Meta-Llama-3-8B_duplication_[(0,4),(27,4)]": "F", # first, last/ blocks / medium
    "Meta-Llama-3-8B_duplication_[(0,5)]": "D", # first/ blocks / big
    "Meta-Llama-3-8B_duplication_[(13,5)]": "C", # middle/ blocks / big
    "Meta-Llama-3-8B_duplication_[(14,3)]": "A", #middle/ blocks / medium
    "Meta-Llama-3-8B_duplication_[(15,1),(16,1)]": "H", #middle/ in_place / small / true
    "Meta-Llama-3-8B_duplication_[(16,2)]": "B", # middle/ in_place / medium
    "Meta-Llama-3-8B_duplication_[(27,5)]": "E", # last/ blocks / big
    "Meta-Llama-3-70B": "original",
    "Meta-Llama-3-8B_duplication_[(26,3),(29,3)]": "R",  # last / blocks / medium / true
    "gemma-2-2b_duplication_[(24,2)]": "P",              # last / blocks / small
    "gemma-2-2b_duplication_[(23,1),(24,1),(25,1),(26,1),(27,1)]": "S",  # last / in_place / small / true
    "gemma-2-2b_duplication_[(22,3),(25,3)]": "R",       # last / blocks / medium / true
    "gemma-2-2b_duplication_[(21,7)]": "Q",              # last / blocks / big
    "gemma-2-2b_duplication_[(14,2),(16,2),(18,2)]": "M",# middle / blocks / small / true
    "gemma-2-2b_duplication_[(14,2),(14,2),(16,2),(16,2)]": "O",  # middle / blocks / small / true
    "gemma-2-2b_duplication_[(13,1),(14,1),(15,1)]": "L",# middle / in_place / small / true
    "gemma-2-2b_duplication_[(12,3),(15,3)]": "N",       # middle / blocks / medium / true
    "gemma-2-2b_duplication_[(12,1),(13,1),(14,1),(15,1)]": "K",  # middle / in_place / small / true
    "Meta-Llama-3-8B_duplication_[(28,2)]": "P",         # last / blocks / small
    "Meta-Llama-3-8B_duplication_[(27,1),(28,1),(29,1),(30,1),(31,1)]": "S",  # last / in_place / small / true
    "Meta-Llama-3-8B_duplication_[(25,7)]": "Q",         # last / blocks / big
    "Meta-Llama-3-8B_duplication_[(16,2),(18,2),(20,2)]": "M",  # middle / blocks / small / true
    "Meta-Llama-3-8B_duplication_[(16,2),(16,2),(18,2),(18,2)]": "O",  # middle / blocks / small / true
    "Meta-Llama-3-8B_duplication_[(15,1),(16,1),(17,1)]": "L",  # middle / in_place / small / true
    "Meta-Llama-3-8B_duplication_[(14,3),(17,3)]": "N",  # middle / blocks / medium / true
    "Meta-Llama-3-8B_duplication_[(14,1),(15,1),(16,1),(17,1)]": "K"  # middle / in_place / small / true
}

# Configuration attribute mapping
configuration_attribute_map = {
    "A": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "medium",
        "continuous": False
    },
    "B": {
        "position": ["middle"],
        "technique": "in place",
        "size": "medium",
        "continuous": False
    },
    "C": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "big",
        "continuous": False
    },
    "D": {
        "position": ["first"],
        "technique": "blocks",
        "size": "big",
        "continuous": False
    },
    "E": {
        "position": ["last"],
        "technique": "blocks",
        "size": "medium",
        "continuous": False
    },
    "F": {
        "position": ["first", "last"],
        "technique": "blocks",
        "size": "medium",
        "continuous": False
    },
    "G": {
        "position": ["first", "last"],
        "technique": "in place",
        "size": "small",
        "continuous": False
    },
    "H": {
        "position": ["middle"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    },
    "I": {
        "position": ["first", "middle", "last"],
        "technique": "in place",
        "size": "small",
        "continuous": False
    },
    "original": {
        "position": [],
        "technique": None,
        "size": None,
        "continuous": None
    },
    "K": {
        "position": ["middle"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    },
    "L": {
        "position": ["middle"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    },
    "M": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "small",
        "continuous": True
    },
    "N": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "medium",
        "continuous": True
    },
    "O": {
        "position": ["middle"],
        "technique": "blocks",
        "size": "small",
        "continuous": True
    },
    "P": {
        "position": ["last"],
        "technique": "blocks",
        "size": "small",
        "continuous": False
    },
    "Q": {
        "position": ["last"],
        "technique": "blocks",
        "size": "big",
        "continuous": False
    },
    "R": {
        "position": ["last"],
        "technique": "blocks",
        "size": "medium",
        "continuous": True
    },
    "S": {
        "position": ["last"],
        "technique": "in place",
        "size": "small",
        "continuous": True
    }
}
