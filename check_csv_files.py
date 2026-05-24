# -*- coding: utf-8 -*-
"""检查 data/ 目录中的 CSV 文件是否可用作真实数据"""
import csv
import os

data_dir = r"C:\Users\A\.qclaw\workspace\DataAgent\data"

files = [
    "Accumulator_Condition.csv",
    "Pump_Leak.csv",
    "System_Failure.csv",
    "Cooler_Condition.csv",
    "Valve_Condition.csv"
]

print("="*60)
print("检查 CSV 文件（可能是 UCI Hydraulic 真实数据）")
print("="*60)

for fn in files:
    fp = os.path.join(data_dir, fn)
    if not os.path.exists(fp):
        print(f"\n[SKIP] {fn} (不存在)")
        continue
    
    print(f"\n文件: {fn}")
    print(f"  大小: {os.path.getsize(fp)//1024} KB")
    
    try:
        with open(fp, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            n_rows = sum(1 for _ in reader)
        
        print(f"  列数: {len(header)}")
        print(f"  行数: {n_rows}")
        print(f"  列名: {header[:5]}")
        print(f"  [OK] 文件有效，可用作真实数据！")
    
    except Exception as e:
        print(f"  [ERR] {e}")

print("\n" + "="*60)
print("结论:")
print("  如果上述文件有效，则 UCI Hydraulic 可以使用真实数据！")
print("="*60)
