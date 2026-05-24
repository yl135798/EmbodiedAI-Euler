# -*- coding: utf-8 -*-
"""
工业数据集自动下载器 v5
- 使用 curl.exe（已验证可访问外网）
- 多镜像源，自动切换
- 支持断点续传
"""

import os
import subprocess
import time
import zipfile
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CURL = "curl.exe"  # Windows 自带，在 System32


def curl_download(urls, out_path, desc="", timeout=300, max_retries=3):
    """用 curl 下载，多源自动切换，支持断点续传"""
    desc = desc or os.path.basename(out_path)
    part = out_path + ".part"

    for attempt in range(max_retries):
        for i, url in enumerate(urls):
            out_tmp = part
            cmd = [
                CURL, "-L", "-k",
                "--connect-timeout", "15",
                "--max-time", str(timeout),
                "--retry", "2",
                "--retry-delay", "3",
                "-w", "HTTP=%{http_code} SIZE=%{size_download}\\n",
            ]

            # 断点续传
            if os.path.exists(part):
                cmd += ["-C", "-"]
                print(f"    [续传] {desc} ({os.path.getsize(part)//1024} KB 已下载)")

            cmd += ["-o", part, url]

            print(f"  [源 {i+1}] {desc}")
            print(f"  URL: {url[:90]}")

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout + 30
                )
                output = result.stdout + result.stderr
                print(f"  {output.strip()[-120:]}")

                if result.returncode == 0 and os.path.exists(part):
                    fsize = os.path.getsize(part)
                    if fsize > 1024:   # 至少 1KB
                        os.replace(part, out_path)
                        print(f"  [OK] {desc} ({fsize//1024} KB)")
                        return True
                    else:
                        print(f"  [WARN] 文件过小 ({fsize} bytes)，可能不是有效数据")
                        os.remove(part)
                else:
                    print(f"  [ERR] 下载失败 (code={result.returncode})")
            except subprocess.TimeoutExpired:
                print(f"  [ERR] 超时 ({timeout}s)")
            except Exception as e:
                print(f"  [ERR] {e}")

            print()  # 源之间空行

        if attempt < max_retries - 1:
            print(f"  [重试] 第 {attempt+2}/{max_retries} 轮...")
            time.sleep(5)

    print(f"  [FAIL] 所有源均失败: {desc}")
    return False


def download_nasa():
    """下载 NASA Turbofan (CMAPSS) 数据集"""
    print("\n" + "="*60)
    print("数据集 1/3: NASA Turbofan (CMAPSS)")
    print("="*60)

    # 多镜像源（按可靠性排序）
    # 来源: Kaggle API（需登录，无法直接 curl）
    # 备用: 各大学/研究者 GitHub 镜像

    # 方案1: 下载 ZIP 包（包含全部4个子集）
    zip_urls = [
        # GitHub 镜像（多个备份）
        "https://github.com/nicolaschen1/Remaining-Useful-Life-Prediction-Using-Deep-Learning/raw/refs/heads/master/data/CMAPSS.zip",
        "https://github.com/AlertTiger/CMAPSS/raw/refs/heads/master/data/CMAPSS.zip",
        # Google Drive 直链（需要处理确认页面）
    ]
    zip_path = os.path.join(DATA_DIR, "CMAPSS.zip")

    if curl_download(zip_urls, zip_path, "CMAPSS.zip", timeout=600):
        print("  解压 ZIP...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 只解压 .txt 数据文件
                txt_files = [f for f in zf.namelist()
                             if f.endswith('.txt') and ('train' in f or 'test' in f or 'RUL' in f)]
                print(f"  找到 {len(txt_files)} 个数据文件，解压中...")
                zf.extractall(DATA_DIR)
            print("  [OK] 解压完成")
            return True
        except Exception as e:
            print(f"  [ERR] 解压失败: {e}")

    # 方案2: 逐文件下载（从多个 GitHub 镜像）
    print("\n  [备选] 逐文件下载...")
    repos = [
        "https://raw.githubusercontent.com/nicolaschen1/Remaining-Useful-Life-Prediction-Using-Deep-Learning/master/data",
        "https://raw.githubusercontent.com/AlertTiger/CMAPSS/master/data",
        "https://raw.githubusercontent.com/blue-chen/remaining-useful-life-prediction-based-on-cmapss/master/data",
    ]

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    ok = 0
    total = 12

    for sub in subsets:
        for ftype in ["train", "test", "RUL"]:
            fname = f"{ftype}_{sub}.txt"
            fpath = os.path.join(DATA_DIR, fname)
            urls = [f"{repo}/{fname}" for repo in repos]

            if curl_download(urls, fpath, fname, timeout=60):
                ok += 1
            time.sleep(0.5)  # 避免请求过快

    print(f"\n  NASA: {ok}/{total} 文件下载成功")
    return ok >= 6


def download_uci_hydraulic():
    """下载 UCI Hydraulic System 数据集"""
    print("\n" + "="*60)
    print("数据集 2/3: UCI Hydraulic System")
    print("="*60)

    # UCI 官方需要 Session，无法直接 curl
    # 备用: GitHub 镜像 / 直接 CSV 下载

    # 方案1: 从 UCI ML 镜像（部分大学镜像可用）
    mirror_urls = [
        # UCI 官方（需 cookie，可能失败）
        "https://archive.ics.uci.edu/static/public/438/condition+monitoring+of+hydraulic+systems.zip",
        # Kaggle 镜像
        "https://www.kaggle.com/api/v1/datasets/download/mihirchintawar/hydraulic-system-condition-monitoring",
        # 直接 CSV（如果 UCI 开放了直接下载）
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00343/Cooler_Condition.csv",
    ]

    # 实际上最可靠的方式：从 GitHub 找镜像
    # 搜索 "UCI hydraulic system dataset GitHub"
    github_urls = [
        "https://github.com/HenryJundong/HydraulicSystem/raw/refs/heads/master/data/Cooler_Condition.csv",
        "https://github.com/mihir1001/Hydraulic-System-Dataset/raw/refs/heads/main/data/Cooler_Condition.csv",
    ]

    print("  [方案1] 尝试从 GitHub 镜像下载 CSV 文件...")

    csv_files = [
        "Cooler_Condition.csv",
        "Valve_Condition.csv",
        "Pump_Leak.csv",
        "Accumulator_Condition.csv",
        "System_Failure.csv",
    ]

    ok = 0
    for csv in csv_files:
        urls = [f"https://github.com/HenryJundong/HydraulicSystem/raw/refs/heads/master/data/{csv}"]
        fpath = os.path.join(DATA_DIR, csv)
        if curl_download(urls, fpath, csv, timeout=60):
            ok += 1
        time.sleep(0.5)

    if ok >= 3:
        print(f"\n  [OK] UCI Hydraulic: {ok}/5 CSV 文件下载成功")
        return True

    print(f"\n  [WARN] 仅下载 {ok}/5 个文件，将使用模拟数据")
    return False


def download_secom():
    """下载 SECOM 半导体数据集（已验证 curl 可访问）"""
    print("\n" + "="*60)
    print("数据集 3/3: SECOM Semiconductor")
    print("="*60)

    data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

    data_path  = os.path.join(DATA_DIR, "secom.data")
    label_path = os.path.join(DATA_DIR, "secom_labels.data")

    ok1 = curl_download([data_url], data_path, "secom.data", timeout=120)
    time.sleep(1)
    ok2 = curl_download([label_url], label_path, "secom_labels.data", timeout=60)

    if ok1 and ok2:
        print("\n  [OK] SECOM 下载完成")
        return True
    else:
        print("\n  [WARN] SECOM 下载失败，将使用模拟数据")
        return False


def main():
    print("="*60)
    print("  DataAgent - 工业数据集自动下载器 v5")
    print("  使用 curl.exe + 多镜像源")
    print("="*60)

    results = {}
    results["nasa"]  = download_nasa()
    results["uci"]    = download_uci_hydraulic()
    results["secom"]  = download_secom()

    # 总结
    print("\n" + "="*60)
    print("  下载结果")
    print("="*60)
    for k, v in results.items():
        s = "[OK] 成功" if v else "[FAIL] 失败（将使用模拟数据）"
        print(f"  {k.upper():10s} {s}")

    print(f"\n  数据目录: {DATA_DIR}")
    print("  目录内容:")
    if os.path.exists(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            fp = os.path.join(DATA_DIR, f)
            if os.path.isfile(fp) and not f.endswith(".part"):
                sz = os.path.getsize(fp) / 1024
                print(f"    {f:45s} {sz:8.1f} KB")
            elif os.path.isdir(fp):
                print(f"    {f:45s} <DIR>")

    print("\n  [OK] 下载流程完成！")
    print("="*60)


if __name__ == "__main__":
    main()
