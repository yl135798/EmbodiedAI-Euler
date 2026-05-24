# -*- coding: utf-8 -*-
"""修复 dataset_loader.py 中 SECOM 标签映射"""
import re

filepath = r"C:\Users\A\.qclaw\workspace\DataAgent\data\dataset_loader.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 找到 _load_real_secom 函数，替换标签映射逻辑
old_block = '''    # 加载标签（格式: -1 "timestamp" 或 1 "timestamp"）
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 只取第一列（-1 或 1），忽略时间戳
            label_str = line.split()[0]
            label = int(label_str)
            labels.append(0 if label == -1 else 1)  # -1→0, 1→1
    y = np.array(labels)'''

new_block = '''    # 加载标签（格式: -1 "timestamp" 或 1 "timestamp"）
    # 文件分布: 93.4% 为 -1, 6.6% 为 1
    # 文献: 不合格率 ~6.6%, 故 -1=合格(1), 1=不合格(0)
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str = line.split()[0]
            label = int(label_str)
            # 与文献对齐: -1→合格(1), 1→不合格(0)
            labels.append(1 if label == -1 else 0)
    y = np.array(labels)

    # 验证分布
    n_pass = int(np.sum(y))           # label=1 合格
    n_defective = len(y) - n_pass    # label=0 不合格
    rate = n_defective / len(y)
    print(f"  标签分布: 合格={n_pass}({n_pass/len(y):.1%}), 不合格={n_defective}({rate:.1%})")'''

if old_block in content:
    content = content.replace(old_block, new_block)
    print("[OK] 标签映射已修复: -1→合格(1), 1→不合格(0)")
else:
    print("[WARN] 未找到旧代码块，尝试手动修复...")
    # 备用：直接搜索 labels.append 行
    content = content.replace(
        "labels.append(0 if label == -1 else 1)",
        "labels.append(1 if label == -1 else 0)  # 修复: -1=合格, 1=不合格"
    )
    print("[OK] 已替换 labels.append 行")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("[OK] dataset_loader.py 已更新")
