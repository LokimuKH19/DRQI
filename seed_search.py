import json
import subprocess
import copy

# Basic settings
BASE_CONFIG_PATH = "CONFIG_DIRICHLET.json"
DYNAMIC_CONFIG_PATH = "config_dynamic.json"
MAIN_SCRIPT = "DRQI_Laplace2d.py"
algorithms = ["DRQI", "DRM", "IPMNN"]
seed_range = range(28, 33)  # 25 - 34

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

