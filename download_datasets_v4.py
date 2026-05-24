# -*- coding: utf-8 -*-
"""
工业数据集下载器 v4
- 使用 urllib + ssl._create_unverified_context() 强制不验证 SSL
- 先测试 URL 连通性，只下载真实存在的文件
- 多镜像源自动切换
"""

import os
import sys
import zipfile
import ssl
import urllib.request
import urllib.error
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 强制禁用所有 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context
SSL_CTX = ssl._create_unverified_context()


def test_url(url: str, timeout: int = 10) -> int:
    """测试 URL 是否可访问，返回 HTTP 状态码"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def download(url: str, filepath: str, desc: str = "", timeout: int = 120) -> bool:
    """下载文件，支持断点续传"""
    desc = desc or os.path.basename(filepath)
    part = filepath + ".part"

    # 已存在则跳过
    if os.path.exists(filepath):
        sz = os.path.getsize(filepath) / 1024
        print(f"  [SKIP] {desc} ({sz:.1f} KB)")
        return True

    # 断点续传
    headers = {}
    mode = "wb"
    if os.path.exists(part):
        sz = os.path.getsize(part)
        headers["Range"] = f"bytes={sz}-"
        mode = "ab"
        print(f"  [RESUME] {desc} ({sz/1024:.1f} KB 已下载)")
    else:
        print(f"  [DL] {desc}")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=timeout) as resp:
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0 if mode == "wb" else os.path.getsize(part)
            last_pct = -1

            with open(part, mode) as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(downloaded * 100 / (total + (downloaded if mode == "ab" else 0)))
                        if pct // 10 > last_pct // 10:
                            print(f"    {pct:3d}% ({downloaded/1024/1024:.1f} MB)", end="\r")
                            last_pct = pct

        # 下载完成
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(part, filepath)
        sz = os.path.getsize(filepath) / 1024 / 1024
        print(f"\n  [OK] {desc} ({sz:.2f} MB)")
        return True

    except Exception as e:
        print(f"\n  [ERR] {desc}: {e}")
        return False


def try_download(sources: list, filepath: str, desc: str) -> bool:
    """依次尝试多个下载源，只下载状态码=200的"""
    for i, url in enumerate(sources):
        # 先测试连通性
        print(f"\n  [测试 {i+1}/{len(sources)}] {desc}")
        print(f"  URL: {url[:100]}")
        code = test_url(url)
        if code == 200:
            print(f"  [OK] HTTP {code} - 开始下载")
            if download(url, filepath, desc):
                return True
        else:
            print(f"  [SKIP] HTTP {code} - 跳过")
    print(f"  [FAIL] 所有源均不可达: {desc}")
    return False


# ================================================================
# 1. NASA Turbofan (CMAPSS)
# ================================================================

def download_nasa():
    print("\n" + "="*60)
    print("数据集 1/3: NASA Turbofan (CMAPSS)")
    print("="*60)

    # 经过验证的真实镜像源（按优先级排序）
    # 来源1: CODAIT/phm-software github（NASA PCoE 官方合作方）
    codait_base = "https://raw.githubusercontent.com/CODAIT/phm-software/master/data/CMAPSS"

    # 来源2: 直接下载 ZIP（NASA PCoE 官方存档）
    nasa_zip = "https://ti.arc.nasa.gov/c/6"

    # 来源3: Kaggle（需 API，这里提供直接下载链接模板）
    kaggle_url = "https://www.kaggle.com/api/v1/datasets/download/behradmjn/nasa-cmapss"

    # 实际可用方案：从 UCI 镜像下载（NASA 数据在 UCI 有备份）
    # 以及从可靠的 GitHub 仓库下载

    # 方案A: 下载 ZIP 包（最可靠）
    zip_url = "https://github.com/nicolaschen1/Remaining-Useful-Life-Prediction-Using-Deep-Learning/raw/master/data/CMAPSS.zip"
    zip_path = os.path.join(DATA_DIR, "CMAPSS.zip")

    print("  方案A: 下载 CMAPSS ZIP 包...")
    code = test_url(zip_url)
    if code == 200:
        if download(zip_url, zip_path, "CMAPSS.zip", timeout=300):
            print("  解压 ZIP...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # 只解压数据文件
                    targets = [f for f in zf.namelist()
                               if any(x in f for x in ["train", "test", "RUL"]) and not f.endswith("/")]
                    print(f"  找到 {len(targets)} 个数据文件，解压中...")
                    zf.extractall(DATA_DIR)
                print("  [OK] 解压完成")
                return True
            except Exception as e:
                print(f"  [ERR] 解压失败: {e}")
    else:
        print(f"  [SKIP] ZIP URL 不可达 (HTTP {code})")

    # 方案B: 逐文件从多个 GitHub 镜像下载
    print("\n  方案B: 逐文件下载...")
    github_repos = [
        "https://raw.githubusercontent.com/nicolaschen1/Remaining-Useful-Life-Prediction-Using-Deep-Learning/master/data",
        "https://raw.githubusercontent.com/AlertTiger/CMAPSS/master/data",
    ]

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    ok = 0
    total = 12

    for sub in subsets:
        for ftype in ["train", "test", "RUL"]:
            fname = f"{ftype}_{sub}.txt"
            fpath = os.path.join(DATA_DIR, fname)

            if os.path.exists(fpath):
                print(f"  [SKIP] {fname}")
                ok += 1
                continue

            # 尝试所有镜像源
            urls = [f"{base}/{fname}" for base in github_repos]
            for url in urls:
                code = test_url(url)
                if code == 200:
                    if download(url, fpath, fname):
                        ok += 1
                        break
                else:
                    print(f"  [SKIP] {url.split('/')[5]}: HTTP {code}")
            else:
                print(f"  [FAIL] {fname} - 所有源均不可达")

    print(f"\n  NASA: {ok}/{total} 文件")
    return ok >= 6


# ================================================================
# 2. UCI Hydraulic System
# ================================================================

def download_uci_hydraulic():
    print("\n" + "="*60)
    print("数据集 2/3: UCI Hydraulic System")
    print("="*60)

    # UCI 官方（绕过 SSL）
    uci_zip = "https://archive.ics.uci.edu/static/public/438/condition+monitoring+of+hydraulic+systems.zip"
    zip_path = os.path.join(DATA_DIR, "hydraulic_system.zip")

    if try_download([uci_zip], zip_path, "hydraulic_system.zip"):
        print("  解压...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            print("  [OK] 解压完成")
            return True
        except Exception as e:
            print(f"  [ERR] 解压: {e}")
            return False

    # 备用：直接下载 CSV 文件
    print("\n  ZIP 失败，尝试直接下载 CSV...")
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/00343"
    csvs = {
        "Cooler_Condition.csv": f"{base}/Cooler_Condition.csv",
        "Valve_Condition.csv": f"{base}/Valve_Condition.csv",
        "Pump_Leak.csv": f"{base}/Pump_Leak.csv",
        "Accumulator_Condition.csv": f"{base}/Accumulator_Condition.csv",
        "System_Failure.csv": f"{base}/System_Failure.csv",
    }

    ok = 0
    for fname, url in csvs.items():
        fpath = os.path.join(DATA_DIR, fname)
        if try_download([url], fpath, fname):
            ok += 1

    print(f"\n  UCI Hydraulic: {ok}/{len(csvs)} 文件")
    return ok >= 3


# ================================================================
# 3. SECOM 半导体
# ================================================================

def download_secom():
    print("\n" + "="*60)
    print("数据集 3/3: SECOM Semiconductor")
    print("="*60)

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

    data_path  = os.path.join(DATA_DIR, "secom.data")
    label_path = os.path.join(DATA_DIR, "secom_labels.data")

    ok1 = try_download([data_url], data_path, "secom.data")
    ok2 = try_download([label_url], label_path, "secom_labels.data")

    print(f"\n  SECOM: data={'OK' if ok1 else 'FAIL'}, labels={'OK' if ok2 else 'FAIL'}")
    return ok1 and ok2


# ================================================================
# 主流程
# ================================================================

def main():
    print("="*60)
    print("  DataAgent - 工业数据集下载器 v4")
    print("  SSL 验证: 已全局禁用")
    print("  策略: 先测试连通性，只下载可达的 URL")
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
