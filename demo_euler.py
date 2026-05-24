# -*- coding: utf-8 -*-
"""
DataAgent - 欧拉同构版 Demo
核心公式: e^(iθ) = cos(θ) + i·sin(θ)

与旧版对比:
  - 旧版: LogicEngine + EmotionEngine（分离）
  - 新版: EulerCognitiveEngine（统一复平面）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cmath
import time
from datetime import datetime

# -- 欧拉同构认知引擎 ---------------------------------
from cognition.cognitive_core_euler import EulerCognitiveEngine

# -- 感知层 -----------------------------------------
from perception.sensor_fusion import FiveSensePerception, SensorReading

# -- 决策执行层 --------------------------------------
from decision.executor import Executor

# -- 量化层 ------------------------------------------
from quantization.scorer import QuantizationScorer

# -- 数据层 ------------------------------------------
from data.dataset_loader import load_secom, load_air_quality


def make_sensor_reading(modality: str, data: np.ndarray, confidence: float = 0.9) -> SensorReading:
    """构造一个感官读数"""
    return SensorReading(
        modality=modality,
        timestamp=time.time(),
        data=data,
        confidence=confidence,
        metadata={"sensor_id": f"{modality}_sensor_01"},
    )


def run_one_sample_euler(engine, perception, executor, scorer,
                          vec: np.ndarray, label: float, modality: str = "visual") -> dict:
    """
    使用欧拉引擎运行单个样本

    流程:
        1. 感知融合
        2. 逻辑推理（实部）
        3. 情绪推断（虚部）
        4. 欧拉融合: z = r · e^(iθ)
        5. 行动计算: (resource, power)
        6. 量化评分
    """
    # 1. 感知
    vec = vec.astype(np.float32)
    vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
    reading = make_sensor_reading(modality, vec, confidence=0.9)
    perception.ingest(reading)
    fused = perception.fuse(window_sec=1.0)

    # 2. 逻辑推理（实部）
    logic_result = engine.reason(fused)

    # 3. 情绪推断（虚部）
    emotion_result = engine.infer_emotion(fused)

    # 4. 欧拉融合: z = r · e^(iθ)
    z = engine.blend(logic_result, emotion_result)

    # 5. 行动计算
    resource, power = engine.compute_action(z)

    # 6. 量化评分
    score_result = engine.score(z)

    return {
        "complex_state": z,
        "magnitude": abs(z),
        "phase": cmath.phase(z),
        "resource": resource,
        "power": power,
        "aggregated_score": score_result["aggregated_score"],
    }


def demo_secom_euler():
    """Demo: SECOM + 欧拉引擎"""
    print("\n" + "="*60)
    print("Demo (欧拉版): SECOM 半导体制造")
    print("="*60)

    # 1. 加载数据
    (X_train, y_train), (X_test, y_test), meta = load_secom()
    print(f"  样本数: {len(X_train) + len(X_test)}")
    print(f"  特征维度: {meta['n_features']}")
    print(f"  不合格率: {meta['defective_rate']:.1%}")

    # 2. 初始化引擎
    engine = EulerCognitiveEngine(state_dim=512, n_emotions=8)
    perception = FiveSensePerception()
    executor = Executor()
    scorer = QuantizationScorer()

    # 3. 测试
    n_test = min(50, len(X_test))
    results = []

    print(f"\n[测试] 运行 {n_test} 个样本...")
    for i in range(n_test):
        sample = X_test[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_test[i])

        vec = sample[:512]
        result = run_one_sample_euler(
            engine, perception, executor, scorer,
            vec, label, modality="visual"
        )
        results.append(result)

    # 4. 统计
    scores = [r["aggregated_score"] for r in results]
    magnitudes = [r["magnitude"] for r in results]
    phases = [r["phase"] for r in results]

    print(f"\n  平均评分: {np.mean(scores):.4f}")
    print(f"  平均模长 |z|: {np.mean(magnitudes):.4f}")
    print(f"  平均幅角 arg(z): {np.mean(phases):.4f} rad")

    # 5. 不合格 vs 合格
    defective_scores = [r["aggregated_score"] for r, lb in zip(results, y_test[:n_test]) if lb == 1]
    normal_scores = [r["aggregated_score"] for r, lb in zip(results, y_test[:n_test]) if lb == 0]

    if defective_scores:
        print(f"  不合格样本平均分: {np.mean(defective_scores):.4f}")
    if normal_scores:
        print(f"  合格样本平均分:   {np.mean(normal_scores):.4f}")

    return {
        "mean_score": np.mean(scores),
        "mean_magnitude": np.mean(magnitudes),
        "mean_phase": np.mean(phases),
    }


def demo_air_quality_euler():
    """Demo: Air Quality + 欧拉引擎"""
    print("\n" + "="*60)
    print("Demo (欧拉版): Air Quality 气体传感器")
    print("="*60)

    # 1. 加载数据
    (X_train, y_train), (X_test, y_test), meta = load_air_quality()
    print(f"  样本数: {len(X_train) + len(X_test)}")
    print(f"  传感器数: {meta['n_features']}")
    print(f"  缺失值率: {meta['missing_rate']:.1%}")

    # 2. 初始化引擎
    engine = EulerCognitiveEngine(state_dim=512, n_emotions=8)
    perception = FiveSensePerception()
    executor = Executor()
    scorer = QuantizationScorer()

    # 3. 测试
    n_test = min(50, len(X_test))
    results = []

    print(f"\n[测试] 运行 {n_test} 个样本...")
    for i in range(n_test):
        sample = X_test[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_test[i])

        vec = sample[:512]
        result = run_one_sample_euler(
            engine, perception, executor, scorer,
            vec, label, modality="olfactory"
        )
        results.append(result)

    # 4. 统计
    scores = [r["aggregated_score"] for r in results]
    magnitudes = [r["magnitude"] for r in results]
    phases = [r["phase"] for r in results]

    print(f"\n  平均评分: {np.mean(scores):.4f}")
    print(f"  平均模长 |z|: {np.mean(magnitudes):.4f}")
    print(f"  平均幅角 arg(z): {np.mean(phases):.4f} rad")

    return {
        "mean_score": np.mean(scores),
        "mean_magnitude": np.mean(magnitudes),
        "mean_phase": np.mean(phases),
    }


def main():
    print("="*60)
    print("  DataAgent - 欧拉同构版 Demo")
    print("  核心公式: e^(iθ) = cos(θ) + i·sin(θ)")
    print("="*60)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Demo 1: SECOM（半导体）
    try:
        results["secom"] = demo_secom_euler()
    except Exception as e:
        print(f"  [SECOM Demo 失败]: {e}")
        import traceback
        traceback.print_exc()

    # Demo 2: Air Quality（气体传感器）
    try:
        results["air_quality"] = demo_air_quality_euler()
    except Exception as e:
        print(f"  [Air Quality Demo 失败]: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print("\n" + "="*60)
    print("  总结: 欧拉同构架构测试结果")
    print("="*60)

    if "secom" in results:
        r = results["secom"]
        print(f"  [SECOM]          平均评分: {r['mean_score']:.4f}")
        print(f"                    平均模长: {r['mean_magnitude']:.4f}")

    if "air_quality" in results:
        r = results["air_quality"]
        print(f"  [Air Quality]    平均评分: {r['mean_score']:.4f}")
        print(f"                    平均模长: {r['mean_magnitude']:.4f}")

    print("\n  [OK] 欧拉同构版 Demo 完成！")
    print("="*60)


if __name__ == "__main__":
    main()
