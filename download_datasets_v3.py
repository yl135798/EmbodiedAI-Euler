# -*- coding: utf-8 -*-
"""
工业数据集下载器 v3
使用 curl --insecure 绕过 SSL 证书问题（Windows 自带 curl）
"""

import os
import sys
import zipfile
import subprocess
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def curl_download(url: str, filepath: str, desc: str = "") -> bool:
    """用 curl --insecure 下载文件（绕过 SSL）"""
    desc = desc or os.path.basename(filepath)
    part = filepath + ".part"

    # 已存在则跳过
    if os.path.exists(filepath):
        sz = os.path.getsize(filepath) / 1024
        print(f"  [SKIP] {desc} ({sz:.1f} KB)")
        return True

    print(f"  [DL] {desc}")
    print(f"       {url[:90]}")

    # 断点续传
    resume = os.path.exists(part)
    cmd = ["curl", "--insecure", "--fail", "--show-error", "--progress-bar",
           "--connect-timeout", "30", "--max-time", "300",
           "-o", part]
    if resume:
        cmd += ["-C", "-"]
        print(f"       断点续传: {os.path.getsize(part)/1024:.1f} KB")

    cmd += [url]

    try:
        result = subprocess.run(cmd, check=False, capture_output=True,
                               text=True, timeout=320)
        if result.returncode == 0 and os.path.exists(part):
            os.rename(part, filepath)
            sz = os.path.getsize(filepath) / 1024 / 1024
            print(f"  [OK] {desc} ({sz:.2f} MB)")
            return True
        else:
            err = result.stderr[:200] if result.stderr else f"code={result.returncode}"
            print(f"  [ERR] {desc}: {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [ERR] {desc}: 超时")
        return False
    except Exception as e:
        print(f"  [ERR] {desc}: {e}")
        return False


def try_sources(sources: list, filepath: str, desc: str) -> bool:
    for i, url in enumerate(sources):
        print(f"\n  [源 {i+1}/{len(sources)}] {desc}")
        if curl_download(url, filepath, desc):
            return True
    print(f"  [FAIL] 所有源均失败: {desc}")
    return False


# ================================================================
# 1. NASA Turbofan (CMAPSS)
# ================================================================

def download_nasa():
    print("\n" + "="*60)
    print("数据集 1/3: NASA Turbofan (CMAPSS)")
    print("="*60)

    # 验证过的 GitHub 镜像源
    sources = {
        "FD001": [
            "https://raw.githubusercontent.com/kennethleungty/NASA-C-MAPSS-Jet-Engine-Predictive-Maintenance/master/data/train_FD001.txt",
            "https://raw.githubusercontent.com/blue-chen/remaining-useful-life-prediction-based-on-cmapss/master/data/train_FD001.txt",
        ],
        "FD002": [
            "https://raw.githubusercontent.com/kennethleungty/NASA-C-MAPSS-Jet-Engine-Predictive-Maintenance/master/data/train_FD002.txt",
        ],
        "FD003": [
            "https://raw.githubusercontent.com/kennethleungty/NASA-C-MAPSS-Jet-Engine-Predictive-Maintenance/master/data/train_FD003.txt",
        ],
        "FD004": [
            "https://raw.githubusercontent.com/kennethleungty/NASA-C-MAPSS-Jet-Engine-Predictive-Maintenance/master/data/train_FD004.txt",
        ],
    }

    # 尝试直接下载 .zip 包（一次搞定）
    zip_url = "https://github.com/kennethleungty/NASA-C-MAPSS-Jet-Engine-Predictive-Maintenance/archive/refs/heads/master.zip"
    zip_path = os.path.join(DATA_DIR, "nasa_cmapss_master.zip")

    print("  尝试下载完整 ZIP 包（推荐）...")
    if curl_download(zip_url, zip_path, "nasa_cmapss_master.zip"):
        print("  解压 ZIP...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 只解压 data/ 目录
                data_files = [f for f in zf.namelist()
                              if f.startswith("NASA-C-MAPSS") and "/data/" in f and not f.endswith("/")]
                print(f"  找到 {len(data_files)} 个数据文件")
                zf.extractall(DATA_DIR)
            print("  [OK] 解压完成")
            return True
        except Exception as e:
            print(f"  [ERR] 解压失败: {e}")
            return False

    # ZIP 失败，逐文件下载
    print("\n  ZIP 失败，逐文件下载...")
    ok = 0
    total = 12  # 4子集 × 3文件
    for sub, urls in sources.items():
        for ftype in ["train", "test", "RUL"]:
            fname = f"{ftype}_{sub}.txt"
            fpath = os.path.join(DATA_DIR, fname)
            # 构造对应文件的 URL（替换 train 为 test/RUL）
            file_urls = [u.replace("train_", f"{ftype.lower()}_") for u in urls]
            if try_sources(file_urls, fpath, fname):
                ok += 1
    print(f"\n  NASA: {ok}/{total} 文件")
    return ok >= 6


# ================================================================
# 2. UCI Hydraulic System
# ================================================================

def download_uci_hydraulic():
    print("\n" + "="*60)
    print("数据集 2/3: UCI Hydraulic System")
    print("="*60)

    zip_url = "https://archive.ics.uci.edu/static/public/438/condition+monitoring+of+hydraulic+systems.zip"
    zip_path = os.path.join(DATA_DIR, "hydraulic_system.zip")

    if try_sources([zip_url], zip_path, "hydraulic_system.zip"):
        print("  解压...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            print("  [OK] 解压完成")
            return True
        except Exception as e:
            print(f"  [ERR] 解压: {e}")
    else:
        print("  [WARN] 下载失败，将使用模拟数据")
    return False


# ================================================================
# 3. SECOM 半导体
# ================================================================

def download_secom():
    print("\n" + "="*60)
    print("数据集 3/3: SECOM Semiconductor")
    print("="*60)

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

    data_path = os.path.join(DATA_DIR, "secom.data")
    label_path = os.path.join(DATA_DIR, "secom_labels.data")

    ok1 = try_sources([data_url], data_path, "secom.data")
    ok2 = try_sources([label_url], label_path, "secom_labels.data")

    if ok1 and ok2:
        print("\n  [OK] SECOM 下载完成")
        return True
    else:
        print("\n  [WARN] 下载失败，将使用模拟数据")
        return False


# ================================================================
# 主流程
# ================================================================

def main():
    print("="*60)
    print("  DataAgent - 工业数据集下载器 v3")
    print("  使用 curl --insecure 绕过 SSL 证书问题")
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
        s = "[OK] 成功" if v else "[WARN] 失败（使用模拟数据）"
        print(f"  {k.upper():10s} {s}")

    print(f"\n  数据目录: {DATA_DIR}")
    print("  目录内容:")
    if os.path.exists(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            fp = os.path.join(DATA_DIR, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp) / 1024
                print(f"    {f:45s} {sz:8.1f} KB")
            elif os.path.isdir(fp):
                print(f"    {f:45s} <DIR>")

    print("\n  [OK] 下载流程完成！")
    print("="*60)


if __name__ == "__main__":
    main()
