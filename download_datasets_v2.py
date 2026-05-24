# -*- coding: utf-8 -*-
"""
工业数据集下载器 v2
- 使用 urllib + ssl._create_unverified_context() 绕过 SSL
- 多镜像源自动切换
"""

import os
import sys
import zipfile
import ssl
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 创建不验证 SSL 的上下文（全局生效）
_ssl_ctx = ssl._create_unverified_context()
ssl._create_default_https_context = ssl._create_unverified_context


def download(url: str, filepath: str, desc: str = "") -> bool:
    desc = desc or os.path.basename(filepath)
    part = filepath + ".part"

    # 断点续传
    headers = {}
    mode = "wb"
    if os.path.exists(part):
        size = os.path.getsize(part)
        headers["Range"] = f"bytes={size}-"
        mode = "ab"
        print(f"  [RESUME] {desc} ({size/1024:.1f} KB 已下载)")
    elif os.path.exists(filepath):
        print(f"  [SKIP] {desc} ({os.path.getsize(filepath)/1024:.1f} KB)")
        return True

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=_ssl_ctx, timeout=60) as resp:
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

        # 下载完成，去掉 .part
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(part, filepath)
        sz = os.path.getsize(filepath)
        print(f"\n  [OK] {desc} ({sz/1024/1024:.2f} MB)")
        return True

    except Exception as e:
        print(f"\n  [ERR] {desc}: {e}")
        return False


def try_sources(sources: list, filepath: str, desc: str) -> bool:
    for i, url in enumerate(sources):
        print(f"\n  源 {i+1}/{len(sources)}: {desc}")
        print(f"  URL: {url[:90]}")
        if download(url, filepath, desc):
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

    # GitHub 上可靠的镜像（直接 raw 文件）
    base = "https://raw.githubusercontent.com/alexhag/CMAPSS/master/data"

    files = {}
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        files[sub] = (
            f"{base}/train_{sub}.txt",
            f"{base}/test_{sub}.txt",
            f"{base}/RUL_{sub}.txt",
        )

    # 备用镜像
    backup_base = "https://raw.githubusercontent.com/biswajitsamanta/NASA-CMAPSS/main/data"

    ok = 0
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for i, suffix in enumerate(["train", "test", "RUL"]):
            fname = f"{suffix}_{sub}.txt"
            # 本地路径
            fpath = os.path.join(DATA_DIR, fname)
            # 主源
            url1 = f"{base}/{fname}"
            # 备用源
            url2 = f"{backup_base}/{fname}"
            if try_sources([url1, url2], fpath, fname):
                ok += 1

    print(f"\n  NASA 完成: {ok}/12 文件")
    return ok >= 6


# ================================================================
# 2. UCI Hydraulic System
# ================================================================

def download_uci_hydraulic():
    print("\n" + "="*60)
    print("数据集 2/3: UCI Hydraulic System")
    print("="*60)

    # UCI 直接下载 ZIP
    uci_zip = "https://archive.ics.uci.edu/static/public/438/condition+monitoring+of+hydraulic+systems.zip"

    # 镜像: Mendeley Data (公开镜像)
    mendeley = "https://data.mendeley.com/datasets/n6g9bj73sr/1/files/9344b3a0-8a6d-4920-83e9-02f5db0260d5/ConditionMonitoringHydraulic.zip"

    # Kaggle 镜像（需登录，跳过）
    # 直接尝试 UCI + Mendeley
    zip_path = os.path.join(DATA_DIR, "hydraulic_system.zip")

    if try_sources([uci_zip, mendeley], zip_path, "hydraulic_system.zip"):
        print("  解压...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            print("  [OK] 解压完成")
            return True
        except Exception as e:
            print(f"  [ERR] 解压: {e}")
            return False

    # 如果 ZIP 失败，尝试直接下载 CSV 文件
    print("\n  ZIP 失败，尝试直接下载 CSV...")
    csv_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/00343"
    csv_files = ["Cooler_Condition.csv", "Valve_Condition.csv", "Pump_Leak.csv",
                 "Accumulator_Condition.csv", "System_Failure.csv"]

    ok = 0
    for cf in csv_files:
        fpath = os.path.join(DATA_DIR, cf)
        if try_sources([f"{csv_base}/{cf}"], fpath, cf):
            ok += 1

    print(f"\n  UCI Hydraulic 完成: {ok}/{len(csv_files)} 文件")
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

    data_path = os.path.join(DATA_DIR, "secom.data")
    label_path = os.path.join(DATA_DIR, "secom_labels.data")

    ok1 = try_sources([data_url], data_path, "secom.data")
    ok2 = try_sources([label_url], label_path, "secom_labels.data")

    print(f"\n  SECOM 完成: data={'OK' if ok1 else 'FAIL'}, labels={'OK' if ok2 else 'FAIL'}")
    return ok1 and ok2


# ================================================================
# 主流程
# ================================================================

def main():
    print("="*60)
    print("  DataAgent - 工业数据集下载器 v2")
    print("  SSL 验证: 已禁用（兼容证书过期）")
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
        s = "[OK] 成功" if v else "[WARN] 失败(将使用模拟数据)"
        print(f"  {k.upper():10s} {s}")

    print(f"\n  数据目录: {DATA_DIR}")
    print("  目录内容:")
    if os.path.exists(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            fp = os.path.join(DATA_DIR, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp) / 1024
                print(f"    {f:45s} {sz:8.1f} KB")

    print("\n  [OK] 下载流程完成")
    print("="*60)


if __name__ == "__main__":
    main()
