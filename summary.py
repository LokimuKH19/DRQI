import os
import re
import math
import pandas as pd
from collections import defaultdict

ROOT_DIR = "./logs"
OUTPUT_FILE = "summary_results.xlsx"

ALGORITHMS = ["DRM", "IPMNN", "DRQI"]

# --------------------------------------------------
# 解析单个 .logs 文件
# --------------------------------------------------
def parse_log_file(filepath):
    aee = None
    msee = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Lambda_Error" in line:
                aee = float(line.split(":")[-1].strip())
            elif "U Loss" in line:
                msee = float(line.split(":")[-1].strip())
    return aee, msee


# --------------------------------------------------
# 计算 log10 均值与标准差
# --------------------------------------------------
def log_stats(values):
    logs = [math.log10(v) for v in values if v > 0]
    if len(logs) == 0:
        return None, None

    mu = sum(logs) / len(logs)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in logs) / len(logs))
    return mu, sigma


# --------------------------------------------------
# 格式化为 10^{μ±σ}
# --------------------------------------------------
def format_log_result(mu, sigma):
    if mu is None or sigma is None:
        return "N/A"
    return rf"{mu:.2f} ± {sigma:.2f}"


# --------------------------------------------------
# 主逻辑
# --------------------------------------------------
def main():
    rows = []

    # 遍历 ./logs/ 下所有 3.1* 文件夹
    for folder in sorted(os.listdir(ROOT_DIR)):
        if not folder.startswith("3.1"):
            continue

        folder_path = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # 数据结构
        seeds = set()
        data = {
            alg: {"AEE": [], "MSEE": []}
            for alg in ALGORITHMS
        }

        # 扫描 log 文件
        for fname in os.listdir(folder_path):
            if not fname.endswith(".log"):
                continue

            # 示例：DRM_LP_1D_SEED28_20251216_212537.logs
            match = re.match(r"(DRM|IPMNN|DRQI).*SEED(\d+)", fname)
            if not match:
                continue

            alg, seed = match.group(1), int(match.group(2))
            seeds.add(seed)

            filepath = os.path.join(folder_path, fname)
            aee, msee = parse_log_file(filepath)

            if aee is not None:
                data[alg]["AEE"].append(aee)
            if msee is not None:
                data[alg]["MSEE"].append(msee)
            print(folder, fname, alg, seed, aee, msee)
        # 统计并写一行
        row = {
            "Folder": folder,
            "Seeds": "{" + ",".join(map(str, sorted(seeds))) + "}",
        }

        for alg in ALGORITHMS:
            mu, sigma = log_stats(data[alg]["AEE"])
            row[f"{alg} AEE"] = format_log_result(mu, sigma)

            mu, sigma = log_stats(data[alg]["MSEE"])
            row[f"{alg} MSEE"] = format_log_result(mu, sigma)

        rows.append(row)

    # 写 Excel
    df = pd.DataFrame(rows, columns=[
        "Folder", "Seeds",
        "DRM AEE", "DRM MSEE",
        "IPMNN AEE", "IPMNN MSEE",
        "DRQI AEE", "DRQI MSEE",
    ])

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
