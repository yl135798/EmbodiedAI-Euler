# -*- coding: utf-8 -*-
"""
工业数据集下载器 v9 - CWRU + IMS 轴承数据集
- CWRU: Case Western Reserve University 轴承故障数据集（经典真实数据）
- IMS: NASA IMS 轴承退化数据集（经典真实数据）
"""

import os
import sys
import time
import zipfile
import urllib.request
import urllib.error
import ssl
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 禁用 SSL 验证
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def log(msg):
    print(msg, flush=True)
    log_path = os.path.join(DATA_DIR, "_download_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def download_file(urls, out_path, desc="", timeout=120):
    """用 urllib 下载，严格超时"""
    desc = desc or os.path.basename(out_path)
    part = out_path + ".part"
    
    log(f"\n[DL] {desc} (timeout={timeout}s)")
    
    for i, url in enumerate(urls):
        log(f"  [源 {i+1}/{len(urls)}] {url[:90]}")
        
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
                http_code = resp.getcode() or 200
                total_size = int(resp.headers.get("Content-Length", 0))
                
                log(f"    HTTP {http_code}, Size={total_size//1024 if total_size else '?'} KB")
                
                if http_code != 200:
                    log(f"    [ERR] HTTP {http_code}")
                    continue
                
                # 下载
                downloaded = 0
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (1024*1024) == 0:
                            log(f"    ... {downloaded//1024//1024} MB")
                
                # 验证
                if downloaded > 1024:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                    os.replace(part, out_path)
                    log(f"  [OK] {desc} ({downloaded//1024} KB)")
                    return True
                else:
                    log(f"    [ERR] 文件过小 ({downloaded} bytes)")
                    if os.path.exists(part):
                        os.remove(part)
        
        except urllib.error.URLError as e:
            log(f"    [ERR] URL 错误: {e.reason}")
        except ssl.SSLError as e:
            log(f"    [ERR] SSL 错误: {e}")
        except Exception as e:
            log(f"    [ERR] {type(e).__name__}: {e}")
    
    log(f"  [FAIL] 所有源均失败: {desc}")
    return False


def download_cwru():
    """
    下载 CWRU 轴承数据集
    来源: GitHub 镜像（多个可用）
    """
    log("="*60)
    log("数据集 1/2: CWRU 轴承故障数据集（真实）")
    log("="*60)
    
    # 检查是否已有
    cwru_dir = os.path.join(DATA_DIR, "CWRU")
    if os.path.exists(cwru_dir):
        n_files = len([f for f in os.listdir(cwru_dir) if f.endswith(".mat")])
        if n_files >= 4:
            log(f"  [SKIP] 已存在 {n_files} 个 .mat 文件")
            return True
    
    # 多个 GitHub 镜像源
    urls = [
        "https://github.com/ck37/R-prediction/raw/master/data-raw/CWRU.zip",
        "https://github.com/yotsuna/CWRU-Bearing-Dataset/archive/refs/heads/master.zip",
        "https://github.com/zeeshann/rolling-element-bearing-fault-diagnosis/raw/master/data/12k_Drive_End_B007_0.mat",
    ]
    
    out = os.path.join(DATA_DIR, "CWRU.zip")
    if download_file(urls, out, "CWRU.zip", timeout=180):
        try:
            with zipfile.ZipFile(out, 'r') as zf:
                zf.extractall(DATA_DIR)
            log("  [OK] 解压完成")
            return True
        except Exception as e:
            log(f"  [ERR] 解压失败: {e}")
            return False
    else:
        # 尝试直接下载单个 .mat 文件（更可靠）
        log("  [重试] 直接下载 .mat 文件...")
        mat_urls = [
            "https://s3.amazonaws.com/tt-submission/cwru/12k_Drive_End_B007_0.mat",
            "https://raw.githubusercontent.com/ck37/R-prediction/master/data-raw/12k_Drive_End_B007_0.mat",
        ]
        mat_out = os.path.join(DATA_DIR, "12k_Drive_End_B007_0.mat")
        if download_file(mat_urls, mat_out, "B007_0.mat", timeout=60):
            log("  [OK] 下载单个 .mat 文件成功（可用于测试）")
            return True
    
    log("  [WARN] CWRU 下载失败，将使用模拟数据")
    return False


def download_ims():
    """
    下载 IMS 轴承数据集（NASA）
    来源: GitHub 镜像
    """
    log("="*60)
    log("数据集 2/2: IMS 轴承退化数据集（真实）")
    log("="*60)
    
    # 检查是否已有
    ims_dir = os.path.join(DATA_DIR, "IMS")
    if os.path.exists(ims_dir):
        n_files = len([f for f in os.listdir(ims_dir) if f.endswith(".mat") or f.endswith(".txt")])
        if n_files >= 2:
            log(f"  [SKIP] 已存在 {n_files} 个文件")
            return True
    
    # IMS 数据集镜像（多个源）
    urls = [
        "https://github.com/karlapal/IMS-Bearing-Data/archive/refs/heads/master.zip",
        "https://github.com/hust512/PHM-Data/raw/master/IMS.zip",
    ]
    
    out = os.path.join(DATA_DIR, "IMS.zip")
    if download_file(urls, out, "IMS.zip", timeout=180):
        try:
            with zipfile.ZipFile(out, 'r') as zf:
                zf.extractall(DATA_DIR)
            log("  [OK] 解压完成")
            return True
        except Exception as e:
            log(f"  [ERR] 解压失败: {e}")
            return False
    else:
        log("  [WARN] IMS 下载失败，将使用模拟数据")
        return False


def main():
    log("="*60)
    log("  DataAgent - CWRU + IMS 数据集下载器")
    log("  真实工业数据集（轴承故障预测）")
    log("="*60)
    
    results = {}
    results["cwru"] = download_cwru()
    results["ims"]  = download_ims()
    
    # 总结
    log("\n" + "="*60)
    log("  下载结果")
    log("="*60)
    for k, v in results.items():
        s = "[OK] 成功" if v else "[FAIL] 失败（将使用模拟数据）"
        log(f"  {k.upper():10s} {s}")
    
    # 显示 data/ 目录内容
    log(f"\n  数据目录: {DATA_DIR}")
    log("  目录内容:")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith("_"):
            continue
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp) and not f.endswith(".part"):
            sz = os.path.getsize(fp)
            unit = "MB" if sz > 1024*1024 else "KB"
            sz_val = sz/1024/1024 if sz > 1024*1024 else sz/1024
            log(f"    {f:45s} {sz_val:8.1f} {unit}")
        elif os.path.isdir(fp):
            n = len([x for x in os.listdir(fp) if not x.startswith("_")])
            log(f"    {f:45s} <DIR> ({n} files)")
    
    log("\n" + "="*60)
    log("  下载流程完成!")
    log("="*60)


if __name__ == "__main__":
    main()
