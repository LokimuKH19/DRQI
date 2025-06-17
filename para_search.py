import json
import itertools
import subprocess
import copy
import os

BASE_CONFIG_PATH = "CONFIG_DIRICHLET.json"
DYNAMIC_CONFIG_PATH = "config_dynamic.json"
MAIN_SCRIPT = "DRQI_Laplace2d.py"    # locate the main process

# Hyperparameters waiting to be changed, fluctuation 0.2 means ±20%
search_keys = ["learning_rate", "omega", "points_per_dimension"]
fluctuation = 0.2

# Basic Configuration
with open(BASE_CONFIG_PATH, "r") as f:
    base_config = json.load(f)

# fluctuation 0.2 means ±20% for all parameters
search_values = {}
for key in search_keys:
    val = base_config[key]
    search_values[key] = [round(val * (1 - fluctuation), 8), val, round(val * (1 + fluctuation), 8)]

# Generate all combination dictionaries
combinations = list(itertools.product(*search_values.values()))
print(f"[INFO] Total combinations (including base config): {len(combinations)}")

# Search combinations (16 combinations is ok)
for i, combo in enumerate(combinations):
    new_config = copy.deepcopy(base_config)
    combo_dict = dict(zip(search_keys, combo))
    for k, v in combo_dict.items():
        new_config[k] = v

    # Write into the dynamic config
    with open(DYNAMIC_CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)

    print(f"\n[Run {i + 1}/{len(combinations)}] Params: {combo_dict}")
    subprocess.run(["python", MAIN_SCRIPT])
    if i == 5:
        break
