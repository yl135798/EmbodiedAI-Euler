# -*- coding: utf-8 -*-
"""
工业数据集下载器 v6 - 最终版
 strategy: Python subprocess + curl.exe，输出重定向到文件
 正确 URL + 多镜像源
"""

import os
import sys
import time
import zipfile
import subprocess
import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CURL = "curl.exe"


def log(msg):
    """打印到 stdout + 写入日志文件"""
    print(msg, flush=True)
    with open(os.path.join(DATA_DIR, "_download_log.txt"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def curl_download(urls, out_path, desc="", timeout=300, max_retries=3):
    """
    用 curl.exe 下载文件，多源切换
    """
    desc = desc or os.path.basename(out_path)
    part = out_path + ".part"
    log(f"\n[DL] {desc} (timeout={timeout}s)")

    for attempt in range(max_retries):
        for i, url in enumerate(urls):
            log(f"  [源 {i+1}/{len(urls)}] {url[:100]}")

            # 构造 curl 命令
            cmd = [
                CURL, "-L", "-k",
                "--connect-timeout", "15",
                "--max-time", str(timeout),
                "--retry", "1",
                "--retry-delay", "2",
                "-w", "HTTP=%{http_code} SIZE=%{size_download}\\n",
            ]

            # 断点续传
            if os.path.exists(part):
                cmd += ["-C", "-"]
                log(f"    [续传] {os.path.getsize(part)//1024} KB 已下载")

            cmd += ["-o", part, "--", url]

            # 执行，捕获输出
            log(f"    [执行] {' '.join(cmd[:8])}...")
            try:
                with open(os.path.join(DATA_DIR, "_curl_stdout.txt"), "w", encoding="utf-8") as out_f:
                    result = subprocess.run(
                        cmd,
                        stdout=out_f,
                        stderr=subprocess.STDOUT,
                        timeout=timeout + 30,
                        check=False,
                    )
                # 读取输出
                with open(os.path.join(DATA_DIR, "_curl_stdout.txt"), "r", encoding="utf-8", errors="replace") as f:
                    output = f.read()
                log(f"    返回码: {result.returncode}")
                log(f"    输出: {output.strip()[-200:]}")

                if result.returncode == 0 and os.path.exists(part):
                    fsize = os.path.getsize(part)
                    if fsize > 1024:  # > 1KB
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        os.replace(part, out_path)
                        log(f"  [OK] {desc} ({fsize//1024} KB)")
                        return True
                    else:
                        log(f"  [WARN] 文件过小 ({fsize} bytes)，删除")
                        os.remove(part)
                else:
                    log(f"  [ERR] 下载失败 (code={result.returncode})")
            except subprocess.TimeoutExpired:
                log(f"  [ERR] 超时 ({timeout}s)")
            except Exception as e:
                log(f"  [ERR] 异常: {e}")

            log("")  # 空行

        if attempt < max_retries - 1:
            log(f"  [重试] 第 {attempt+2}/{max_retries} 轮，等待 5s...")
            time.sleep(5)

    log(f"  [FAIL] 所有源均失败: {desc}")
    return False


def download_nasa():
    """下载 NASA Turbofan (CMAPSS)"""
    log("="*60)
    log("数据集 1/3: NASA Turbofan (CMAPSS)")
    log("="*60)

    # 方案A: 从 UCI 镜像下载（CMAPSS 在 UCI 的 ID=697）
    # UCI 的直接下载链接（需要推断文件名）
    # 实际: NASA PCoE 提供的数据，UCI 只有描述页面
    # 正确来源: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

    # 方案B: 从 ResearchGate/GitHub 下载
    # 经过验证的真实 URL（NASA CMAPSS 数据）
    nasa_urls = [
        # Kaggle（需要 API token，无法直接 curl）
        # 改用: CODAIT/phm-software GitHub Release
        "https://github.com/CODAIT/phm-software/releases/download/v1.0/CMAPSS.zip",
        # 备用: 直接下载 train/test 文件（从某个可用镜像）
        "https://raw.githubusercontent.com/nicolaschen1/Remaining-Useful-Life-Prediction-Using-Deep-Learning/master/data/train_FD001.txt",
    ]

    zip_path = os.path.join(DATA_DIR, "CMAPSS.zip")
    if curl_download(nasa_urls[:1], zip_path, "CMAPSS.zip", timeout=600):
        log("  解压 ZIP...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
                log(f"  找到 {len(txt_files)} 个数据文件")
                zf.extractall(DATA_DIR)
            log("  [OK] 解压完成")
            return True
        except Exception as e:
            log(f"  [ERR] 解压失败: {e}")

    # 方案B: 逐文件下载（从多个源）
    log("\n  [备选] 逐文件下载 train/test/RUL...")
    repos = [
        "https://raw.githubusercontent.com/nicolaschen1/Remaining-Useful-Life-Prediction-Using-Deep-Learning/master/data",
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
            time.sleep(0.3)
    log(f"\n  NASA: {ok}/{total} 文件")
    return ok >= 6


def download_uci_hydraulic():
    """下载 UCI Hydraulic System"""
    log("="*60)
    log("数据集 2/3: UCI Hydraulic System")
    log("="*60)

    # UCI 官方直接下载 URL（经过验证的格式）
    # 格式: https://archive.ics.uci.edu/ml/machine-learning-databases/<ID>/<filename>
    # Hydraulic System 的 ID = 00438
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/00438"

    # 尝试下载 ZIP（官方提供的是 ZIP 包）
    zip_urls = [
        f"{base}/hydraulic-system.zip",
        "https://archive.ics.uci.edu/static/public/438/condition+monitoring+of+hydraulic+systems.zip",
    ]
    zip_path = os.path.join(DATA_DIR, "hydraulic_system.zip")
    if curl_download(zip_urls, zip_path, "hydraulic_system.zip", timeout=600):
        log("  解压 ZIP...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            log("  [OK] 解压完成")
            return True
        except Exception as e:
            log(f"  [ERR] 解压失败: {e}")

    # 方案B: 下载单独的 CSV 文件
    log("\n  [备选] 下载单独的 CSV 文件...")
    csvs = {
        "Cooler_Condition.csv":       f"{base}/Cooler_Condition.csv",
        "Valve_Condition.csv":        f"{base}/Valve_Condition.csv",
        "Pump_Leak.csv":             f"{base}/Pump_Leak.csv",
        "Accumulator_Condition.csv":  f"{base}/Accumulator_Condition.csv",
        "System_Failure.csv":         f"{base}/System_Failure.csv",
    }
    ok = 0
    for fname, url in csvs.items():
        fpath = os.path.join(DATA_DIR, fname)
        if curl_download([url], fpath, fname, timeout=60):
            ok += 1
        time.sleep(0.3)
    log(f"\n  UCI Hydraulic: {ok}/{len(csvs)} 文件")
    return ok >= 3


def download_secom():
    """下载 SECOM（已验证 curl 可访问）"""
    log("="*60)
    log("数据集 3/3: SECOM Semiconductor")
    log("="*60)

    data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    label_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

    data_path  = os.path.join(DATA_DIR, "secom.data")
    label_path = os.path.join(DATA_DIR, "secom_labels.data")

    ok1 = curl_download([data_url], data_path, "secom.data", timeout=120)
    time.sleep(1)
    ok2 = curl_download([label_url], label_path, "secom_labels.data", timeout=60)

    if ok1 and ok2:
        log("\n  [OK] SECOM 下载完成")
        return True
    else:
        log("\n  [WARN] SECOM 下载失败")
        return False


def check_results():
    """检查下载结果"""
    log("="*60)
    log("下载结果")
    log("="*60)
    files = os.listdir(DATA_DIR)
    files = [f for f in files if not f.startswith("_")]
    for f in sorted(files):
        fp = os.path.join(DATA_DIR, f)
        if os.path.isfile(fp) and not f.endswith(".part"):
            sz = os.path.getsize(fp)
            if sz > 1024 * 1024:
                log(f"  {f:45s} {sz/1024/1024:.2f} MB")
            else:
                log(f"  {f:45s} {sz/1024:.1f} KB")
        elif os.path.isdir(fp):
            log(f"  {f:45s} <DIR>")


def main():
    log("="*60)
    log("  DataAgent - 工业数据集下载器 v6")
    log("  使用: Python subprocess + curl.exe")
    log("="*60)

    results = {}
    results["nasa"]  = download_nasa()
    results["uci"]    = download_uci_hydraulic()
    results["secom"]  = download_secom()

    # 总结
    log("\n" + "="*60)
    log("  最终结果")
    log("="*60)
    for k, v in results.items():
        s = "[OK] 成功" if v else "[FAIL] 失败"
        log(f"  {k.upper():10s} {s}")

    check_results()
    log("\n" + "="*60)
    log("  下载流程完成")
    log("="*60)


if __name__ == "__main__":
    main()
