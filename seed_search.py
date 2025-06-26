import json
import subprocess
import copy

# Basic settings
BASE_CONFIG_PATH = "CONFIG_PERIODIC.json"     # Select the basic configurations for different problem
DYNAMIC_CONFIG_PATH = "config_dynamic.json"    # DO NOT CHANGE
MAIN_SCRIPT = "DRQI_FokkerPlank2d.py"  # "DRQI_Laplace2d.py"    # change the problems here
algorithms = ["DRQI"]#"IPMNN", "DRM",]
seed_range = range(30, 31)   # select the random seed range

# load base config
with open(BASE_CONFIG_PATH, "r") as f:
    base_config = json.load(f)

# main process
for algo in algorithms:
    print(f"\n=== Testing Algorithm: {algo} ===")
    for seed in seed_range:
        new_config = copy.deepcopy(base_config)
        new_config["algorithm"] = algo
        new_config["seed"] = seed

        # dynamic config dumped
        with open(DYNAMIC_CONFIG_PATH, "w") as f:
            json.dump(new_config, f, indent=2)

        print(f"[Running] Algo: {algo}, Seed: {seed}")
        subprocess.run(["python", MAIN_SCRIPT])

