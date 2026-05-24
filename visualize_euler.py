# -*- coding: utf-8 -*-
"""
欧拉同构架构可视化报告
- 复平面轨迹图
- 模长 |z| 分布
- 幅角 θ 分布
- 新旧版本对比
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import cmath

# 设置中文字体
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
except:
    font = FontProperties()

# -- 数据层 ------------------------------------------
from data.dataset_loader import load_secom, load_air_quality

# -- 欧拉引擎 ----------------------------------------
from cognition.cognitive_core_euler import EulerCognitiveEngine

# -- 感知层 -----------------------------------------
from perception.sensor_fusion import FiveSensePerception, SensorReading

# -- 决策执行层 --------------------------------------
from decision.executor import Executor

# -- 量化层 ------------------------------------------
from quantization.scorer import QuantizationScorer


def make_sensor_reading(modality: str, data: np.ndarray, confidence: float = 0.9):
    """构造感官读数"""
    return SensorReading(
        modality=modality,
        timestamp=time.time(),
        data=data,
        confidence=confidence,
        metadata={"sensor_id": f"{modality}_sensor_01"},
    )


def run_samples_euler(dataset_name: str, X_test, y_test, modality: str, n_samples: int = 50):
    """
    运行样本并收集欧拉状态

    返回:
        trajectories: 复数轨迹列表
        scores: 评分列表
        magnitudes: 模长列表
        phases: 幅角列表
    """
    import time

    engine = EulerCognitiveEngine(state_dim=512, n_emotions=8)
    perception = FiveSensePerception()
    executor = Executor()
    scorer = QuantizationScorer()

    trajectories = []
    scores = []
    magnitudes = []
    phases = []

    for i in range(min(n_samples, len(X_test))):
        sample = X_test[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_test[i])

        # 1. 感知
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
        reading = make_sensor_reading(modality, vec, confidence=0.9)
        perception.ingest(reading)
        fused = perception.fuse(window_sec=1.0)

        # 2. 逻辑推理
        logic_result = engine.reason(fused)

        # 3. 情绪推断
        emotion_result = engine.infer_emotion(fused)

        # 4. 欧拉融合
        z = engine.blend(logic_result, emotion_result)

        # 5. 评分
        score_result = engine.score(z)

        # 6. 收集数据
        trajectories.append(z)
        scores.append(score_result["aggregated_score"])
        magnitudes.append(abs(z))
        phases.append(cmath.phase(z))

    return trajectories, scores, magnitudes, phases


def plot_euler_trajectory(trajectories_secom, trajectories_air, filename="euler_trajectory.png"):
    """绘制复平面轨迹图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("欧拉同构 - 复平面认知轨迹", fontproperties=font, fontsize=16)

    # 1. SECOM 轨迹
    ax = axes[0]
    z_secom = np.array(trajectories_secom)
    ax.scatter(z_secom.real, z_secom.imag, c=np.arange(len(z_secom)),
               cmap="Blues", alpha=0.7, s=30)
    ax.plot(z_secom.real, z_secom.imag, 'b-', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("实部 (Logic, cosθ)", fontproperties=font)
    ax.set_ylabel("虚部 (Emotion, sinθ)", fontproperties=font)
    ax.set_title(f"SECOM 轨迹 (n={len(trajectories_secom)})", fontproperties=font)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # 2. Air Quality 轨迹
    ax = axes[1]
    z_air = np.array(trajectories_air)
    ax.scatter(z_air.real, z_air.imag, c=np.arange(len(z_air)),
               cmap="Reds", alpha=0.7, s=30)
    ax.plot(z_air.real, z_air.imag, 'r-', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("实部 (Logic, cosθ)", fontproperties=font)
    ax.set_ylabel("虚部 (Emotion, sinθ)", fontproperties=font)
    ax.set_title(f"Air Quality 轨迹 (n={len(trajectories_air)})", fontproperties=font)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] 已保存: {out}")
    plt.close()


def plot_magnitude_phase(magnitudes_secom, phases_secom,
                          magnitudes_air, phases_air,
                          filename="magnitude_phase.png"):
    """绘制模长和幅角分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("欧拉同构 - 模长 |z| 与幅角 θ 分布", fontproperties=font, fontsize=16)

    # 1. SECOM 模长分布
    ax = axes[0, 0]
    if np.std(magnitudes_secom) < 1e-6:
        # 所有值相同，用竖线表示
        ax.axvline(x=np.mean(magnitudes_secom), color="blue", linewidth=5, alpha=0.7,
                   label=f"恒定值={np.mean(magnitudes_secom):.3f}")
        ax.set_xlim([0, 1.2])
    else:
        ax.hist(magnitudes_secom, bins=20, color="blue", alpha=0.7, edgecolor="black")
        ax.axvline(x=np.mean(magnitudes_secom), color="red", linestyle="--",
                    label=f"均值={np.mean(magnitudes_secom):.3f}")
    ax.set_xlabel("模长 |z|", fontproperties=font)
    ax.set_ylabel("频数", fontproperties=font)
    ax.set_title(f"SECOM - 模长分布 (n={len(magnitudes_secom)})", fontproperties=font)
    ax.legend(prop=font)
    ax.grid(True, alpha=0.3)

    # 2. Air Quality 模长分布
    ax = axes[0, 1]
    if np.std(magnitudes_air) < 1e-6:
        ax.axvline(x=np.mean(magnitudes_air), color="red", linewidth=5, alpha=0.7,
                    label=f"恒定值={np.mean(magnitudes_air):.3f}")
        ax.set_xlim([0, 1.2])
    else:
        ax.hist(magnitudes_air, bins=20, color="red", alpha=0.7, edgecolor="black")
        ax.axvline(x=np.mean(magnitudes_air), color="blue", linestyle="--",
                    label=f"均值={np.mean(magnitudes_air):.3f}")
    ax.set_xlabel("模长 |z|", fontproperties=font)
    ax.set_ylabel("频数", fontproperties=font)
    ax.set_title(f"Air Quality - 模长分布 (n={len(magnitudes_air)})", fontproperties=font)
    ax.legend(prop=font)
    ax.grid(True, alpha=0.3)

    # 3. SECOM 幅角分布
    ax = axes[1, 0]
    ax.hist(phases_secom, bins=20, color="green", alpha=0.7, edgecolor="black")
    ax.axvline(x=np.mean(phases_secom), color="red", linestyle="--",
                label=f"均值={np.mean(phases_secom):.3f} rad")
    ax.set_xlabel("幅角 θ (rad)", fontproperties=font)
    ax.set_ylabel("频数", fontproperties=font)
    ax.set_title(f"SECOM - 幅角分布 (n={len(phases_secom)})", fontproperties=font)
    ax.legend(prop=font)
    ax.grid(True, alpha=0.3)

    # 4. Air Quality 幅角分布
    ax = axes[1, 1]
    ax.hist(phases_air, bins=20, color="orange", alpha=0.7, edgecolor="black")
    ax.axvline(x=np.mean(phases_air), color="blue", linestyle="--",
                label=f"均值={np.mean(phases_air):.3f} rad")
    ax.set_xlabel("幅角 θ (rad)", fontproperties=font)
    ax.set_ylabel("频数", fontproperties=font)
    ax.set_title(f"Air Quality - 幅角分布 (n={len(phases_air)})", fontproperties=font)
    ax.legend(prop=font)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] 已保存: {out}")
    plt.close()


def plot_score_comparison(scores_secom, scores_air, filename="score_comparison.png"):
    """绘制评分对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["SECOM", "Air Quality"]
    means = [np.mean(scores_secom), np.mean(scores_air)]
    stds = [np.std(scores_secom), np.std(scores_air)]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=10,
                   color=["blue", "red"], alpha=0.7)

    ax.set_ylabel("平均评分", fontproperties=font)
    ax.set_title("欧拉同构 - 数据集评分对比", fontproperties=font, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=font)
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis="y")

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + std + 0.01,
                 f"{mean:.4f}", ha="center", fontproperties=font)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] 已保存: {out}")
    plt.close()


def main():
    print("="*60)
    print("  欧拉同构架构 - 可视化报告生成器")
    print("  核心公式: e^(iθ) = cos(θ) + i·sin(θ)")
    print("="*60)

    # 1. 加载数据
    print("\n[1] 加载真实数据集...")
    (_, _), (X_test_secom, y_test_secom), _ = load_secom()
    (_, _), (X_test_air, y_test_air), _ = load_air_quality()
    print(f"  SECOM 测试集: {len(X_test_secom)} 样本")
    print(f"  Air Quality 测试集: {len(X_test_air)} 样本")

    # 2. 运行欧拉引擎（SECOM）
    print("\n[2] 运行欧拉引擎（SECOM）...")
    traj_secom, scores_secom, mags_secom, phases_secom = run_samples_euler(
        "SECOM", X_test_secom, y_test_secom, modality="visual", n_samples=50
    )
    print(f"  [OK] 完成 {len(traj_secom)} 个样本")
    print(f"  平均评分: {np.mean(scores_secom):.4f}")
    print(f"  平均模长: {np.mean(mags_secom):.4f}")
    print(f"  平均幅角: {np.mean(phases_secom):.4f} rad")

    # 3. 运行欧拉引擎（Air Quality）
    print("\n[3] 运行欧拉引擎（Air Quality）...")
    traj_air, scores_air, mags_air, phases_air = run_samples_euler(
        "Air Quality", X_test_air, y_test_air, modality="olfactory", n_samples=50
    )
    print(f"  [OK] 完成 {len(traj_air)} 个样本")
    print(f"  平均评分: {np.mean(scores_air):.4f}")
    print(f"  平均模长: {np.mean(mags_air):.4f}")
    print(f"  平均幅角: {np.mean(phases_air):.4f} rad")

    # 4. 生成可视化
    print("\n[4] 生成可视化报告...")
    plot_euler_trajectory(traj_secom, traj_air)
    plot_magnitude_phase(mags_secom, phases_secom, mags_air, phases_air)
    plot_score_comparison(scores_secom, scores_air)

    # 5. 总结
    print("\n" + "="*60)
    print("  可视化报告生成完成！")
    print("  输出文件:")
    print("    - euler_trajectory.png    (复平面轨迹)")
    print("    - magnitude_phase.png     (模长/幅角分布)")
    print("    - score_comparison.png   (评分对比)")
    print("="*60)


if __name__ == "__main__":
    import time  # 延迟导入（make_sensor_reading 用到）
    main()
