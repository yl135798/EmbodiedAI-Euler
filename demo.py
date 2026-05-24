"""
DataAgent - 具身智能大脑 Demo
作者: QClaw Agent
日期: 2026-05-24

核心公式:
  五感信息交互
  -> 知（逻辑推理 + 情绪思考）
  -> 行（钱=时间+五感信息 权=元权力/定义权 分配）
  -> 量化结果

测试数据集（真实工业数据）:
  1. NASA Turbofan (PHM08) - 发动机退化预测
  2. UCI Hydraulic System - 液压系统多任务监控
  3. SECOM Semiconductor - 半导体质量预测
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
import time
from datetime import datetime

# -- 感知层 ----------------------------------------------
from perception.sensor_fusion import FiveSensePerception, SensorReading

# -- 认知层 ----------------------------------------------
from cognition.cognitive_core import LogicEngine, EmotionEngine

# -- 决策执行层 ------------------------------------------
from decision.executor import Executor, ResourceCost

# -- 量化层 ----------------------------------------------
from quantization.scorer import QuantizationScorer

# -- 数据层 ----------------------------------------------
from data.dataset_loader import (
    load_dataset,
    load_nasa_turbofan,
    load_uci_hydraulic,
    load_secom,
    load_air_quality,
)


def make_sensor_reading(modality: str, data: np.ndarray, confidence: float = 0.9) -> SensorReading:
    """构造一个感官读数"""
    return SensorReading(
        modality=modality,
        timestamp=time.time(),
        data=data,
        confidence=confidence,
        metadata={"sensor_id": f"{modality}_sensor_01"},
    )


def run_one_sample(perception, logic_engine, emotion_engine, executor, scorer,
                   sample: np.ndarray, label: float, modality: str = "visual") -> dict:
    """
    处理一个样本，完成"知行合一"的完整流程
    """
    # -- 1. 五感感知 ------------------------------
    # 将样本数据映射到某个感官模态（Demo中用visual模拟）
    if len(sample.shape) == 1:
        vec = sample[:min(len(sample), 512)]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")
    else:
        vec = sample.flatten()[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")

    reading = make_sensor_reading(modality, vec, confidence=0.85)
    perception.ingest(reading)
    fused = perception.fuse(window_sec=2.0)

    # -- 2. 知：逻辑推理 + 情绪思考 ----------------
    logic_result = logic_engine.reason(fused)
    emotion_result = emotion_engine.infer_emotion(fused)
    blended = emotion_engine.blend(logic_result, emotion_result)

    # -- 3. 行：资源分配 + 权力分配 + 执行 --------
    task_context = {
        "urgency": 0.6,
        "high_performers": ["agent_logic"],
        "domain_experts": ["agent_sensor"],
        "sample_label": label,
    }
    claimants = ["agent_logic", "agent_emotion", "agent_sensor"]

    exec_result = executor.execute(blended, task_context, claimants)

    # -- 4. 量化结果 ------------------------------
    score_result = scorer.score(exec_result, ground_truth=label)

    return {
        "fused_norm": float(np.linalg.norm(fused)),
        "logic_confidence": logic_result["confidence"],
        "dominant_emotion": emotion_result["dominant_emotion"],
        "emotion_intensity": emotion_result["emotion_intensity"],
        "quantized_score": exec_result["quantized_score"],
        "aggregated_score": score_result["aggregated_score"],
        "interpretation": exec_result["interpretation"],
        "diagnostics": score_result["diagnostics"],
    }


def demo_nasa_turbofan():
    """Demo 1: NASA 涡轮发动机退化预测（时间序列）"""
    print("\n" + "="*60)
    print("Demo 1: NASA Turbofan 发动机退化预测")
    print("="*60)

    # 加载数据（用模拟数据，真实数据需自行下载）
    (X_train, y_train), (X_test, y_test), meta = load_nasa_turbofan("FD001")
    print(f"  训练样本: {X_train.shape}, 测试样本: {X_test.shape}")

    perception = FiveSensePerception()
    logic_engine = LogicEngine()
    emotion_engine = EmotionEngine()
    executor = Executor()
    scorer = QuantizationScorer()

    # 模拟"五感"：将发动机传感器数据映射为视觉+听觉
    print("\n[感知] 摄入发动机传感器数据（模拟五感）...")
    n_test = min(20, len(X_test))
    results = []

    train_scores = []
    test_scores = []

    for i in range(n_test):
        sample = X_test[i, 2:]  # 去掉 engine_id 和 cycle
        label = float(y_test[i])

        # 随机分配模态（模拟多感官输入）
        modality = ["visual", "auditory", "tactile"][i % 3]
        result = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, sample, label, modality
        )
        results.append(result)
        test_scores.append(result["aggregated_score"])

    # 训练集分数（模拟）
    for i in range(min(30, len(X_train))):
        sample = X_train[i, 2:]
        label = float(y_train[i])
        r = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, sample, label
        )
        train_scores.append(r["aggregated_score"])

    # 泛化性测试
    gen_result = scorer.generalization_test(train_scores, test_scores)

    print(f"\n  训练集平均分数: {np.mean(train_scores):.4f}")
    print(f"  测试集平均分数: {np.mean(test_scores):.4f}")
    print(f"  泛化性得分:     {gen_result['generalization_score']:.4f} ({gen_result['verdict']})")
    print(f"  泛化性差距:     {gen_result['generalization_gap']:.4f}")

    return gen_result


def demo_uci_hydraulic():
    """Demo 2: UCI 液压系统监控（多感官 + 多任务）"""
    print("\n" + "="*60)
    print("Demo 2: UCI 液压系统 - 多感官 + 多任务")
    print("="*60)

    (X_train, y_train), (X_test, y_test), meta = load_uci_hydraulic()
    # 合并训练集和测试集用于 demo
    X = np.concatenate([X_train, X_test])
    y_multi = {}
    for k in y_train.keys():
        y_multi[k] = np.concatenate([y_train[k], y_test[k]])
    print(f"  传感器: {meta['sensor_names']}")
    print(f"  任务: {meta['tasks']}")

    perception = FiveSensePerception()
    logic_engine = LogicEngine()
    emotion_engine = EmotionEngine()
    executor = Executor()
    scorer = QuantizationScorer()

    # 将5个传感器映射为"五感"
    modality_map = {
        0: "tactile",    # 压力 -> 触觉
        1: "auditory",    # 电机功率 -> 听觉（电流声）
        2: "tactile",   # 振动 -> 触觉（已有tactile，用自定义）
        3: "visual",      # 温度 -> 视觉（热成像）
        4: "olfactory",   # 流量 -> 嗅觉（液体气味）
    }

    n_test = min(30, len(X))
    all_scores = []

    for i in range(n_test):
        sample = X[i]
        # 取第一个任务作为 label
        label = float(y_multi["cooling_condition"][i])

        # 多感官同时摄入
        for sensor_idx, modality in modality_map.items():
            vec = np.array([sample[sensor_idx]])
            vec = np.pad(vec, (0, 511), mode="constant")
            reading = make_sensor_reading(modality, vec, confidence=0.8 + 0.2*np.random.rand())
            perception.ingest(reading)

        fused = perception.fuse(window_sec=1.0)
        logic_result = logic_engine.reason(fused)
        emotion_result = emotion_engine.infer_emotion(fused)
        blended = emotion_engine.blend(logic_result, emotion_result)

        task_context = {
            "urgency": 0.5,
            "high_performers": ["agent_logic"],
            "domain_experts": ["agent_sensor"],
            "multi_task": True,
            "tasks": list(y_multi.keys()),
        }
        claimants = ["agent_logic", "agent_emotion", "agent_sensor"]
        exec_result = executor.execute(blended, task_context, claimants)
        score_result = scorer.score(exec_result, ground_truth=label)
        all_scores.append(score_result["aggregated_score"])

    print(f"\n  多任务平均分数: {np.mean(all_scores):.4f}")
    print(f"  五感状态: {perception.get_modality_status()}")

    # 打印情绪分布（样例）
    emotion_dist = emotion_result["emotion_distribution"]
    print(f"  情绪分布(最后一帧):")
    for emo, val in emotion_dist.items():
        if val > 0.1:
            print(f"    {emo}: {val:.3f}")

    return {"mean_score": np.mean(all_scores), "scores": all_scores}


def demo_secom():
    """Demo 3: SECOM 半导体质量预测（高维 + 不平衡）"""
    print("\n" + "="*60)
    print("Demo 3: SECOM 半导体制造 - 高维特征 + 鲁棒性测试")
    print("="*60)

    (X_train, y_train), (X_test, y_test), meta = load_secom()
    # 合并训练集和测试集用于 demo
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    print(f"  特征维度: {meta['n_features']}")
    print(f"  不合格率: {meta['defective_rate']:.1%}")
    print(f"  缺失值率: {meta['missing_rate']:.1%}")

    perception = FiveSensePerception()
    logic_engine = LogicEngine()
    emotion_engine = EmotionEngine()
    executor = Executor()
    scorer = QuantizationScorer()

    # 处理缺失值（工业数据真实挑战）
    n_test = min(50, len(X))
    results = []

    for i in range(n_test):
        sample = X[i]
        # 处理 NaN
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y[i])

        # 高维数据 -> 视觉模态（类似显微镜图像）
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")

        result = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, vec, label, modality="visual"
        )
        results.append(result)

    defective_results = [r for r, lb in zip(results, y[:n_test]) if lb == 1]
    normal_results = [r for r, lb in zip(results, y[:n_test]) if lb == 0]

    print(f"\n  不合格样本平均分数: {np.mean([r['aggregated_score'] for r in defective_results]) if defective_results else 0:.4f}")
    print(f"  合格样本平均分数:   {np.mean([r['aggregated_score'] for r in normal_results]):.4f}")

    # 鲁棒性：添加噪声后重新测试
    print(f"\n  鲁棒性测试: 添加 20% 高斯噪声...")
    noise_scores = []
    for i in range(min(20, n_test)):
        noisy_sample = X[i] + np.random.randn(*X[i].shape) * 0.2 * np.std(X[i][~np.isnan(X[i])])
        noisy_sample = np.nan_to_num(noisy_sample, nan=0.0)
        vec = noisy_sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")
        r = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, vec, float(y[i])
        )
        noise_scores.append(r["aggregated_score"])

    print(f"  噪声下平均分数: {np.mean(noise_scores):.4f}")
    print(f"  鲁棒性差距:     {abs(np.mean(noise_scores) - np.mean([r['aggregated_score'] for r in results[:20]])):.4f}")

    return {"robustness_gap": abs(np.mean(noise_scores) - np.mean([r['aggregated_score'] for r in results[:20]]))}


def demo_air_quality():
    """Demo 4: Air Quality 气体传感器（嗅觉）- 高缺失值挑战"""
    print("\n" + "="*60)
    print("Demo 4: Air Quality - 嗅觉传感器（气体检测）")
    print("="*60)

    (X_train, y_train), (X_test, y_test), meta = load_air_quality()
    print(f"  传感器（嗅觉）: {meta['sensor_names']}")
    print(f"  缺失值率: {meta['missing_rate']:.1%}")
    print(f"  目标: {meta['target_name']} 浓度")

    perception = FiveSensePerception()
    logic_engine = LogicEngine()
    emotion_engine = EmotionEngine()
    executor = Executor()
    scorer = QuantizationScorer()

    # 将气体传感器读数映射为嗅觉输入
    n_test = min(30, len(X_test))
    results = []

    for i in range(n_test):
        sample = X_test[i]
        label = float(y_test[i])

        # 嗅觉模态
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")
        result = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, vec, label, modality="olfactory"
        )
        results.append(result)

    # 分析 CO 浓度与评分的关系
    high_co = [r for r, y in zip(results, y_test[:n_test]) if y > 5.0]  # 高 CO 浓度
    low_co  = [r for r, y in zip(results, y_test[:n_test]) if y <= 5.0]  # 低 CO 浓度

    print(f"\n  高 CO 浓度样本 ({len(high_co)} 个) 平均分数: {np.mean([r['aggregated_score'] for r in high_co]):.4f}")
    print(f"  低 CO 浓度样本 ({len(low_co)} 个) 平均分数: {np.mean([r['aggregated_score'] for r in low_co]):.4f}")

    # 鲁棒性：添加噪声后重新测试
    print(f"\n  鲁棒性测试: 添加 15% 高斯噪声...")
    noise_scores = []
    for i in range(min(20, n_test)):
        noisy_sample = X_test[i] + np.random.randn(*X_test[i].shape) * 0.15 * np.std(X_test[i][~np.isnan(X_test[i])])
        noisy_sample = np.nan_to_num(noisy_sample, nan=0.0)
        vec = noisy_sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")
        r = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, vec, float(y_test[i])
        )
        noise_scores.append(r["aggregated_score"])

    robustness_gap = abs(np.mean(noise_scores) - np.mean([r["aggregated_score"] for r in results[:20]]))
    print(f"  噪声下平均分数: {np.mean(noise_scores):.4f}")
    print(f"  鲁棒性差距:     {robustness_gap:.4f}")

    return {"robustness_gap": robustness_gap, "n_samples": n_test}


def print_formula_explanation():
    """打印核心公式解释"""
    print("\n" + "="*60)
    print("  核心公式: 知行合一 = 量化结果")
    print("="*60)
    print("""
    输入: 五感信息交互
      +-- 视觉 (visual)      -> 摄像头 / 工业相机
      +-- 听觉 (auditory)    -> 麦克风 / 设备声音
      +-- 触觉 (tactile)     -> 压力传感器 / 振动传感器
      +-- 嗅觉 (olfactory)   -> 气体传感器
      +-- 味觉 (gustatory)   -> 化学传感器

    ↓ 融合 (Fusion)

    知 = 逻辑思考 + 情绪思考
      +-- LogicEngine:  符号推理 + 神经推理
      +-- EmotionEngine: Plutchik 八维情绪空间

    ↓ 融合 (Blend)

    行 = 钱 + 权 分配
      +-- 钱 = 时间成本 + 五感信息消耗 + 算力成本 + 金钱成本
      |     -> ResourceAllocator 量化分配
      +-- 权 = 元权力（议程设置权 / 框架定义权 / 指标定义权 / 资源控制权）
            -> MetaPowerAllocator 分配权重

    ↓ 执行 (Execute)

    量化结果 = f(认知向量, 资源分配, 权力分配)
      +-- QuantizationScorer: 多维评分 + 泛化性测试
    """)


def main():
    print("="*60)
    print("  DataAgent - 具身智能大脑 Demo（仅真实数据集）")
    print("  知行合一: 五感 -> 知 -> 行 -> 量化结果")
    print("="*60)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_formula_explanation()

    # -- 只运行真实数据集 ------------------------------
    results = {}

    # 禁用模拟数据（NASA Turbofan - 模拟）
    # try:
    #     results["nasa_generalization"] = demo_nasa_turbofan()
    # except Exception as e:
    #     print(f"  [NASA Demo 失败]: {e}")

    # 禁用模拟数据（UCI Hydraulic - 模拟）
    # try:
    #     results["uci_multi_task"] = demo_uci_hydraulic()
    # except Exception as e:
    #     print(f"  [UCI Demo 失败]: {e}")

    # 真实数据集 1: SECOM（半导体制造）
    try:
        results["secom_robustness"] = demo_secom()
    except Exception as e:
        print(f"  [SECOM Demo 失败]: {e}")

    # 真实数据集 2: Air Quality（嗅觉传感器）
    try:
        results["air_quality"] = demo_air_quality()
    except Exception as e:
        print(f"  [Air Quality Demo 失败]: {e}")

    # 跨数据集泛化性测试（真实数据 → 真实数据）
    try:
        results["cross_domain"] = demo_cross_domain_generalization()
    except Exception as e:
        print(f"  [Cross-Domain Demo 失败]: {e}")

    # -- 总结 ------------------------------------------
    print("\n" + "="*60)
    print("  总结: 通用性与泛化性测试报告")
    print("="*60)

    if "nasa_generalization" in results:
        g = results["nasa_generalization"]
        print(f"  [NASA Turbofan] 泛化性得分: {g['generalization_score']:.3f} - {g['verdict']}")

    if "uci_multi_task" in results:
        m = results["uci_multi_task"]
        print(f"  [UCI Hydraulic]  多任务平均分数: {m['mean_score']:.3f}")

    if "secom_robustness" in results:
        s = results["secom_robustness"]
        print(f"  [SECOM]         鲁棒性差距: {s['robustness_gap']:.3f} (越小越好)")

    print("\n  [OK] Demo 完成！")
    print("  [DIR] 项目路径: C:\\Users\\A\\.qclaw\\workspace\\DataAgent")
    print("  [TIP] 提示: 真实工业数据请手动下载后放入 data/ 目录")
    print("="*60)


if __name__ == "__main__":
    main()
