import json
import subprocess
import copy
import os

# 文件路径
BASE_CONFIG_PATH = "CONFIG_DIRICHLET.json"
DYNAMIC_CONFIG_PATH = "config_dynamic.json"
MAIN_SCRIPT = "DRQI_Laplace2d.py"

# 要测试的算法列表
algorithms = ["DRQI", "DRM", "IPMNN"]
seed_range = range(25, 35)  # 25 到 34

# 加载基础配置
with open(BASE_CONFIG_PATH, "r") as f:
    base_config = json.load(f)

# 创建输出目录
os.makedirs("logs", exist_ok=True)

# 循环算法和种子
for algo in algorithms:
    print(f"\n=== Testing Algorithm: {algo} ===")
    for seed in seed_range:
        new_config = copy.deepcopy(base_config)
        new_config["algorithm"] = algo
        new_config["seed"] = seed

        # 保存动态配置
        with open(DYNAMIC_CONFIG_PATH, "w") as f:
            json.dump(new_config, f, indent=2)

        print(f"[Running] Algo: {algo}, Seed: {seed}")
        subprocess.run(["python", MAIN_SCRIPT])

