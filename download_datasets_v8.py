# -*- coding: utf-8 -*-
"""
工业数据集下载器 v8 - 最终稳定版
- 使用 urllib（Python 标准库，无需外部工具）
- 严格超时，永不卡死
- 多镜像源自动切换
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

# 禁用 SSL 验证（UCI 证书过期）
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def log(msg):
    print(msg, flush=True)
    log_path = os.path.join(DATA_DIR, "_download_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def download_file(urls, out_path, desc="", timeout=60, max_retries=3):
    """
    用 urllib 下载，永不卡死（严格超时）
    """
    desc = desc or os.path.basename(out_path)
    part = out_path + ".part"
    
    log(f"\n[DL] {desc} (timeout={timeout}s)")

    for attempt in range(max_retries):
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
                        os.remove(part)

            except urllib.error.URLError as e:
                log(f"    [ERR] URL 错误: {e.reason}")
            except ssl.SSLError as e:
                log(f"    [ERR] SSL 错误: {e}")
            except Exception as e:
                log(f"    [ERR] {type(e).__name__}: {e}")

        if attempt < max_retries - 1:
            log(f"  [重试] 第 {attempt+2}/{max_retries} 轮，等待 3s...")
            time.sleep(3)

    log(f"  [FAIL] 所有源均失败: {desc}")
    return False


def download_air_quality():
    """下载 UCI Air Quality（嗅觉传感器数据）"""
    log("="*60)
    log("数据集 1/4: UCI Air Quality（嗅觉 - 气体传感器）")
    log("="*60)

    # 多镜像源
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
        "https://archive.ics.uci.edu/static/public/360/AirQualityUCI.zip",
    ]
    out = os.path.join(DATA_DIR, "AirQualityUCI.zip")

    if download_file(urls, out, "AirQualityUCI.zip", timeout=120):
        # 解压
        try:
            with zipfile.ZipFile(out, 'r') as zf:
                zf.extractall(DATA_DIR)
            log("  [OK] 解压完成")
            return True
        except Exception as e:
            log(f"  [ERR] 解压失败: {e}")
    return False


def download_ai4i2020():
    """下载 UCI AI4I 2020 Predictive Maintenance（工业预测性维护）"""
    log("="*60)
    log("数据集 2/4: UCI AI4I 2020（工业预测性维护）")
    log("="*60)

    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00460/ai4i2020.csv",
        "https://archive.ics.uci.edu/static/public/460/ai4i2020.csv",
    ]
    out = os.path.join(DATA_DIR, "ai4i2020.csv")

    if download_file(urls, out, "ai4i2020.csv", timeout=120):
        log("  [OK] 下载完成")
        return True
    return False


def download_pamap2():
    """下载 PAMAP2 Physical Activity（多传感器 - 类似五感融合）"""
    log("="*60)
    log("数据集 3/4: PAMAP2（多传感器 - 身体活动监测）")
    log("="*60)

    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        "https://archive.ics.uci.edu/static/public/231/PAMAP2_Dataset.zip",
    ]
    out = os.path.join(DATA_DIR, "PAMAP2_Dataset.zip")

    if download_file(urls, out, "PAMAP2_Dataset.zip", timeout=300):
        try:
            with zipfile.ZipFile(out, 'r') as zf:
                zf.extractall(DATA_DIR)
            log("  [OK] 解压完成")
            return True
        except Exception as e:
            log(f"  [ERR] 解压失败: {e}")
    return False


def download_secom_verify():
    """验证 SECOM 数据完整性"""
    log("="*60)
    log("数据集 4/4: SECOM（验证已有数据）")
    log("="*60)

    data_path  = os.path.join(DATA_DIR, "secom.data")
    label_path = os.path.join(DATA_DIR, "secom_labels.data")

    if os.path.exists(data_path) and os.path.exists(label_path):
        sz1 = os.path.getsize(data_path) // 1024
        sz2 = os.path.getsize(label_path) // 1024
        log(f"  [OK] SECOM 数据完整 ({sz1} KB + {sz2} KB)")
        
        # 验证标签分布
        with open(label_path, "r") as f:
            labels = [l.strip().split()[0] for l in f if l.strip()]
        n_minus1 = labels.count("-1")
        n_plus1  = labels.count("1")
        log(f"  标签分布: -1={n_minus1}({n_minus1/len(labels):.1%}), 1={n_plus1}({n_plus1/len(labels):.1%})")
        return True
    else:
        log("  [WARN] SECOM 数据不完整，重新下载...")
        urls_data  = ["https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"]
        urls_label = ["https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"]
        ok1 = download_file(urls_data, data_path, "secom.data", timeout=120)
        ok2 = download_file(urls_label, label_path, "secom_labels.data", timeout=60)
        return ok1 and ok2


def main():
    log("="*60)
    log("  DataAgent - 工业数据集下载器 v8")
    log("  使用: urllib（Python 标准库，永不卡死）")
    log("="*60)

    results = {}
    results["air_quality"] = download_air_quality()
    results["ai4i2020"]   = download_ai4i2020()
    results["pamap2"]      = download_pamap2()
    results["secom"]        = download_secom_verify()

    # 总结
    log("\n" + "="*60)
    log("  下载结果")
    log("="*60)
    for k, v in results.items():
        s = "[OK] 成功" if v else "[FAIL] 失败"
        log(f"  {k.upper():15s} {s}")

    # 显示目录内容
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
            log(f"    {f:45s} <DIR>")

    log("\n" + "="*60)
    log("  下载流程完成!")
    log("="*60)


if __name__ == "__main__":
    main()
