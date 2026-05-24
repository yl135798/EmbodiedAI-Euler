# -*- coding: utf-8 -*-
"""
可视化报告生成器
- 读取真实数据集（SECOM, Air Quality）
- 生成图表（特征分布、缺失值、评分分布等）
- 保存为 PNG 文件
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 不显示 GUI
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体（Windows）
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
except:
    font = FontProperties()

# -- 数据层 ----------------------------------------------
from data.dataset_loader import load_secom, load_air_quality


def plot_secom_features():
    """SECOM 特征分布图"""
    print("[可视化] SECOM 特征分布...")
    
    (X_train, y_train), (X_test, y_test), meta = load_secom()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    # 计算缺失值率
    missing_rate = np.mean(np.isnan(X), axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("SECOM 半导体制造 - 数据分析", fontproperties=font, fontsize=16)
    
    # 1. 缺失值率分布
    ax = axes[0, 0]
    ax.hist(missing_rate, bins=50, color="skyblue", edgecolor="black")
    ax.set_xlabel("缺失值率", fontproperties=font)
    ax.set_ylabel("特征数", fontproperties=font)
    ax.set_title("特征缺失值率分布", fontproperties=font)
    ax.grid(True, alpha=0.3)
    
    # 2. 标签分布
    ax = axes[0, 1]
    labels = ["合格", "不合格"]
    counts = [np.sum(y == 0), np.sum(y == 1)]
    ax.bar(labels, counts, color=["green", "red"])
    ax.set_ylabel("样本数", fontproperties=font)
    ax.set_title("标签分布（不合格率 6.6%）", fontproperties=font)
    for i, v in enumerate(counts):
        ax.text(i, v + 10, str(v), ha="center", fontproperties=font)
    
    # 3. 前 20 个特征的分布（随机选）
    ax = axes[1, 0]
    np.random.seed(42)
    idx = np.random.choice(X.shape[1], 20, replace=False)
    X_clean = np.nan_to_num(X, nan=0.0)
    ax.boxplot([X_clean[:, i] for i in idx], showfliers=False)
    ax.set_xlabel("特征索引", fontproperties=font)
    ax.set_ylabel("值", fontproperties=font)
    ax.set_title("前 20 个特征分布（去 NaN）", fontproperties=font)
    ax.tick_params(axis="x", rotation=45)
    
    # 4. 缺失值热力图（前 50 个样本 × 前 20 个特征）
    ax = axes[1, 1]
    missing_matrix = np.isnan(X[:50, :20]).astype(int)
    ax.imshow(missing_matrix, cmap="RdYlBu", aspect="auto")
    ax.set_xlabel("特征", fontproperties=font)
    ax.set_ylabel("样本", fontproperties=font)
    ax.set_title("缺失值热力图（前 50 样本 × 前 20 特征）", fontproperties=font)
    
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "secom_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] 已保存: {out}")
    plt.close()


def plot_air_quality():
    """Air Quality 气体传感器数据分析"""
    print("[可视化] Air Quality 气体传感器...")
    
    (X_train, y_train), (X_test, y_test), meta = load_air_quality()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Air Quality - 气体传感器数据分析", fontproperties=font, fontsize=16)
    
    # 1. CO 浓度分布
    ax = axes[0, 0]
    ax.hist(y, bins=50, color="orange", edgecolor="black", alpha=0.7)
    ax.set_xlabel("CO 浓度 (mg/m3)", fontproperties=font)
    ax.set_ylabel("样本数", fontproperties=font)
    ax.set_title("CO 浓度分布", fontproperties=font)
    ax.grid(True, alpha=0.3)
    
    # 2. 传感器读数随时间变化（前 1000 个样本）
    ax = axes[0, 1]
    sensor_names = meta["sensor_names"]
    for i, name in enumerate(sensor_names[:3]):  # 只画前 3 个传感器
        ax.plot(X[:1000, i], label=name, alpha=0.7)
    ax.set_xlabel("样本", fontproperties=font)
    ax.set_ylabel("传感器读数", fontproperties=font)
    ax.set_title("传感器读数随时间变化（前 1000 样本）", fontproperties=font)
    ax.legend(prop=font)
    ax.grid(True, alpha=0.3)
    
    # 3. 缺失值率
    ax = axes[1, 0]
    missing_rate = np.mean(np.isnan(X), axis=0)
    ax.bar(range(len(missing_rate)), missing_rate, color="red", alpha=0.6)
    ax.set_xlabel("传感器索引", fontproperties=font)
    ax.set_ylabel("缺失值率", fontproperties=font)
    ax.set_title(f"缺失值率（平均 {meta['missing_rate']:.1%}）", fontproperties=font)
    ax.grid(True, alpha=0.3)
    
    # 4. CO 浓度 vs 传感器读数（散点图）
    ax = axes[1, 1]
    # 只取非缺失值
    valid = ~np.isnan(X[:, 0]) & ~np.isnan(y)
    ax.scatter(X[valid, 0], y[valid], alpha=0.3, s=5)
    ax.set_xlabel("传感器 1 读数", fontproperties=font)
    ax.set_ylabel("CO 浓度", fontproperties=font)
    ax.set_title("传感器 1 读数 vs CO 浓度", fontproperties=font)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "air_quality_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] 已保存: {out}")
    plt.close()


def plot_cross_domain_gap():
    """跨数据集泛化性差距图"""
    print("[可视化] 跨数据集泛化性差距...")
    
    # 从之前的测试结果
    gaps = [3.6568, 4.6886]
    labels = ["SECOM → Air Quality\n(半导体 → 气体)", "Air Quality → SECOM\n(气体 → 半导体)"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, gaps, color=["blue", "green"], alpha=0.7)
    ax.set_ylabel("泛化性差距（越小越好）", fontproperties=font, fontsize=12)
    ax.set_title("跨数据集泛化性差距（真实数据 → 真实数据）", fontproperties=font, fontsize=14)
    ax.set_ylim([0, max(gaps) * 1.2])
    ax.grid(True, alpha=0.3, axis="y")
    
    # 添加数值标签
    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{gap:.4f}",
            ha="center",
            fontproperties=font,
            fontsize=11
        )
    
    # 添加评判线
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="优秀 (0.1)")
    ax.axhline(y=0.3, color="orange", linestyle="--", alpha=0.5, label="良好 (0.3)")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="一般 (0.5)")
    ax.legend(prop=font)
    
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "cross_domain_gap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] 已保存: {out}")
    plt.close()


def main():
    print("="*60)
    print("  可视化报告生成器")
    print("  真实数据集: SECOM, Air Quality")
    print("="*60)
    
    # 1. SECOM 分析
    try:
        plot_secom_features()
    except Exception as e:
        print(f"  [ERR] SECOM 可视化失败: {e}")
    
    # 2. Air Quality 分析
    try:
        plot_air_quality()
    except Exception as e:
        print(f"  [ERR] Air Quality 可视化失败: {e}")
    
    # 3. 跨数据集泛化性差距
    try:
        plot_cross_domain_gap()
    except Exception as e:
        print(f"  [ERR] 跨数据集可视化失败: {e}")
    
    print("\n" + "="*60)
    print("  可视化报告生成完成！")
    print("  输出文件:")
    print("    - secom_analysis.png")
    print("    - air_quality_analysis.png")
    print("    - cross_domain_gap.png")
    print("="*60)


if __name__ == "__main__":
    main()
