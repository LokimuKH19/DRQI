import os
import re
import json
import subprocess
import re

def extract_timestamp_from_model(model_name):
    """
    DRQI_LP_4D_OMEGA0.5_SEED32_Adam_20251216_203258.pkl
    -> 20251216_203258
    """
    m = re.search(r"(\d{8}_\d{6})(?=\.pkl$)", model_name)
    if not m:
        raise RuntimeError(f"Cannot extract timestamp from model name: {model_name}")
    return m.group(1)


def replace_log_timestamp(log_name, new_timestamp):
    """
    DRQI_LP_4D_SEED32_20250620_121742.log
    -> DRQI_LP_4D_SEED32_20251216_203258.log
    """
    return re.sub(
        r"\d{8}_\d{6}(?=\.log$)",
        new_timestamp,
        log_name
    )


# -----------------------------
# 配置
# -----------------------------
MAIN_SCRIPT = "DRQI_Laplace2d.py"          # 主程序路径
BASE_LOG_DIR = "./logs"                     # 根 logs 目录
DYNAMIC_CONFIG_PATH = "config_dynamic.json" # 临时 JSON 配置文件
MODEL_DIR = "./models"                      # 模型保存路径

# -----------------------------
# 从 log 中解析网络结构
# -----------------------------
def parse_network_structure(log_lines, dimension):
    """
    从 log_lines 中的 Model Architecture 部分解析网络结构
    返回 network_structure 字典
    """
    net_lines = []
    in_net = False
    for line in log_lines:
        if "Model Architecture:" in line:
            in_net = True
            continue
        if in_net:
            if line.strip() == ")":  # 结束
                break
            net_lines.append(line.strip())

    # 提取 Linear 层的 in/out 特征
    layers = []
    for line in net_lines:
        if line.startswith("(") and "Linear" in line:
            in_match = re.search(r'in_features=(\d+)', line)
            out_match = re.search(r'out_features=(\d+)', line)
            if in_match and out_match:
                in_f = int(in_match.group(1))
                out_f = int(out_match.group(1))
                if not layers:  # 第一个 Linear 层
                    layers.append(in_f)
                layers.append(out_f)

    # 返回字典，key 为维度
    network_structure = {str(dimension): layers}
    return network_structure

# -----------------------------
# 从 log 文件解析配置
# -----------------------------
def parse_log(log_path):
    config = {}
    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Algorithm:"):
            config["algorithm"] = line.split(":")[1].strip()
        elif line.startswith("Model Dimension"):
            config["dimension"] = int(re.findall(r'\d+', line)[0])
        elif line.startswith("Learning Rate:"):
            config["learning_rate"] = float(line.split(":")[1].strip())
        elif line.startswith("Unique Parameters:"):
            omega_match = re.search(r"'omega': ([0-9.eE+-]+)", line)
            gamma_match = re.search(r"'gamma': ([0-9.eE+-]+)", line)
            config["omega"] = float(omega_match.group(1)) if omega_match else 0.0
            config["gamma"] = float(gamma_match.group(1)) if gamma_match else 0.0
        elif line.startswith("Number of Samples:"):
            config["points_per_dimension"] = int(line.split(":")[1].strip())/config["dimension"]
        elif line.startswith("Max Epochs:"):
            config["epochs"] = int(line.split(":")[1].strip())
        elif line.startswith("Seed:"):
            config["seed"] = int(line.split(":")[1].strip())
        elif line.startswith("Optimizer:"):
            config["optimizer"] = line.split(":")[1].strip()
        elif line.startswith("Input Sampling Method:"):
            config["sampling_method"] = "QMC" if "QMC" in line else "LHS"
        elif "Early Stopping Threshold (eq_loss < epsilon1)" in line:
            eps1 = float(line.split(":")[1].strip())
            config["epsilon1"] = eps1
        elif "Early Stopping Threshold (eq_loss_k - ep_loss_k-1 < epsilon2)" in line:
            eps2 = float(line.split(":")[1].strip())
            if "epsilon1" in config and config["epsilon1"] != 0:
                config["epsilon2_ratio"] = eps2 / config["epsilon1"]
            else:
                config["epsilon2_ratio"] = 1e-2  # 默认值

    # 解析网络结构
    config["network_structure"] = parse_network_structure(lines, config["dimension"])

    # 填充 optimizer 默认值
    if "optimizer" not in config:
        config["optimizer"] = "Adam"

    return config


# -----------------------------
# 重跑单个 log
# -----------------------------
def rerun_log(log_path, subdir_path):
    print(f"[INFO] Re-running log: {log_path}")

    config = parse_log(log_path)
    with open(DYNAMIC_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    log_name = os.path.basename(log_path)

    # ===== 删除旧 log =====
    os.remove(log_path)

    # ===== 记录运行前模型和 log =====
    before_models = set(os.listdir(MODEL_DIR))
    before_logs = set(os.listdir("./logs"))

    # ===== 运行主程序 =====
    subprocess.run(["python", MAIN_SCRIPT], check=True)

    # ===== 找到新模型 =====
    after_models = set(os.listdir(MODEL_DIR))
    new_models = list(after_models - before_models)
    if len(new_models) != 1:
        raise RuntimeError(f"Expected 1 new model, got {new_models}")

    new_model = new_models[0]
    timestamp = extract_timestamp_from_model(new_model)

    # ===== 找到新 log =====
    after_logs = set(os.listdir("./logs"))
    new_logs = list(after_logs - before_logs)
    if len(new_logs) != 1:
        raise RuntimeError(f"Expected 1 new log, got {new_logs}")

    new_log = new_logs[0]

    # ===== 用模型时间戳替换 log 时间戳 =====
    final_log_name = replace_log_timestamp(log_name, timestamp)

    src = os.path.join("./logs", new_log)
    dst = os.path.join(subdir_path, final_log_name)

    os.replace(src, dst)

    print(f"[INFO] Log renamed correctly → {final_log_name}")
    print(f"[INFO] Matched model → {new_model}")




# -----------------------------
# 批量处理
# -----------------------------
def batch_rerun():
    for subdir in os.listdir(BASE_LOG_DIR):
        subdir_path = os.path.join(BASE_LOG_DIR, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith("3.1.1"):
            print(f"[INFO] Processing directory: {subdir_path}")
            log_files = [f for f in os.listdir(subdir_path) if f.endswith(".log")]
            for log_file in log_files:
                log_path = os.path.join(subdir_path, log_file)
                rerun_log(log_path, subdir_path)

# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    batch_rerun()
