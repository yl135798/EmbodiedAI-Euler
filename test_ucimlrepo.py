# -*- coding: utf-8 -*-
"""最小测试: ucimlrepo 是否能下载 UCI 数据集"""
import sys
import os

DATA_DIR = "C:\\Users\\A\\.qclaw\\workspace\\DataAgent\\data"
os.makedirs(DATA_DIR, exist_ok=True)

print("测试1: 导入 ucimlrepo...")
try:
    from ucimlrepo.fetch_ucirepo import fetch_ucirepo
    print("  [OK] 导入成功")
except Exception as e:
    print(f"  [ERR] 导入失败: {e}")
    sys.exit(1)

print("\n测试2: 下载 UCI 小型数据集 (Iris, ID=53)...")
try:
    # 先用一个小数据集测试
    ds = fetch_ucirepo(53, as_frame=True)  # Iris 数据集
    print(f"  [OK] 下载成功! 形状: {ds.data.features.shape}")
    print(f"  特征: {list(ds.data.features.columns)[:5]}")
    sys.exit(0)
except Exception as e:
    print(f"  [ERR] 下载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试3: 下载 UCI Hydraulic (ID=438)...")
try:
    ds = fetch_ucirepo(438, as_frame=True)
    print(f"  [OK] 下载成功! 形状: {ds.data.features.shape}")
    sys.exit(0)
except Exception as e:
    print(f"  [ERR] 下载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n[结论] ucimlrepo 无法访问 UCI（可能需要浏览器 cookie / 网络限制）")
sys.exit(1)
