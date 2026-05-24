# -*- coding: utf-8 -*-
"""解压 hydraulic_system.zip 并检查内容"""
import zipfile
import os

zp = r"C:\Users\A\.qclaw\workspace\DataAgent\data\hydraulic_system.zip"
out =  r"C:\Users\A\.qclaw\workspace\DataAgent\data"

print("ZIP 内容列表:")
with zipfile.ZipFile(zp, "r") as z:
    names = z.namelist()
    for n in names[:20]:
        info = z.getinfo(n)
        print(f"  {n} ({info.file_size/1024:.1f} KB)")
    print(f"共 {len(names)} 个文件")

    if len(names) > 0:
        print("\n解压中...")
        z.extractall(out)
        print("[OK] 解压完成")
    else:
        print("[WARN] ZIP 为空或已损坏")
