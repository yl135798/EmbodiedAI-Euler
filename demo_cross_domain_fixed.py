# -*- coding: utf-8 -*-
"""
跨数据集泛化性测试（修正版）
- 训练: SECOM（半导体）
- 测试: Air Quality（气体传感器）
- 测试算法能否跨领域泛化
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from perception.sensor_fusion import FiveSensePerception, SensorReading
from cognition.cognitive_core import LogicEngine, EmotionEngine
from decision.executor import Executor
from quantization.scorer import QuantizationScorer
from data.dataset_loader import load_secom, load_air_quality


def run_one_sample(perception, logic_engine, emotion_engine,
                   executor, scorer, vec, label, modality="visual"):
    """运行单个样本（辅助函数）"""
    vec = vec.astype(np.float32)
    vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
    
    reading = SensorReading(
        modality=modality,
        timestamp=time.time(),
        data=vec,
        confidence=0.9,
        metadata={"sensor_id": f"{modality}_sensor_01"},
    )
    perception.ingest(reading)
    fused = perception.fuse(window_sec=1.0)
    logic_result = logic_engine.reason(fused)
    emotion_result = emotion_engine.infer_emotion(fused)
    blended = emotion_engine.blend(logic_result, emotion_result)
    task_context = {
        "urgency": 0.5,
        "high_performers": ["agent_logic"],
        "domain_experts": ["agent_sensor"],
    }
    claimants = ["agent_logic", "agent_emotion", "agent_sensor"]
    exec_result = executor.execute(blended, task_context, claimants)
    score_result = scorer.score(exec_result, ground_truth=label)
    return score_result


def demo_cross_domain_generalization():
    """
    跨数据集泛化性测试
    训练: SECOM（半导体）
    测试: Air Quality（气体传感器）
    """
    print("\n" + "="*60)
    print("跨数据集泛化性测试（真实数据 → 真实数据）")
    print("="*60)
    
    # 1. 加载 SECOM（训练集）
    print("\n[训练集] 加载 SECOM（半导体制造）...")
    (X_train, y_train), (X_test_secom, y_test_secom), meta_secom = load_secom()
    print(f"  SECOM 样本数: {len(X_train)}")
    print(f"  特征维度: {meta_secom['n_features']}")
    print(f"  不合格率: {meta_secom['defective_rate']:.1%}")
    
    # 2. 加载 Air Quality（测试集）
    print("\n[测试集] 加载 Air Quality（嗅觉传感器）...")
    (X_train_aq, y_train_aq), (X_test_aq, y_test_aq), meta_aq = load_air_quality()
    print(f"  Air Quality 样本数: {len(X_test_aq)}")
    print(f"  传感器数: {meta_aq['n_features']}")
    print(f"  缺失值率: {meta_aq['missing_rate']:.1%}")
    
    # 3. 初始化模型
    perception = FiveSensePerception()
    logic_engine = LogicEngine()
    emotion_engine = EmotionEngine()
    executor = Executor()
    scorer = QuantizationScorer()
    
    # 4. 在 SECOM 上训练
    print("\n[训练] 在 SECOM 数据上训练...")
    train_scores = []
    n_train = min(100, len(X_train))
    
    for i in range(n_train):
        sample = X_train[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_train[i])
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
        result = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, vec, label, modality="visual"
        )
        train_scores.append(result["aggregated_score"])
    
    avg_train = np.mean(train_scores)
    print(f"  训练集平均分数: {avg_train:.4f}")
    
    # 5. 在 Air Quality 上测试
    print("\n[测试] 在 Air Quality 数据上测试（跨领域）...")
    test_scores = []
    n_test = min(50, len(X_test_aq))
    
    for i in range(n_test):
        sample = X_test_aq[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_test_aq[i])
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
        result = run_one_sample(
            perception, logic_engine, emotion_engine,
            executor, scorer, vec, label, modality="olfactory"
        )
        test_scores.append(result["aggregated_score"])
    
    avg_test = np.mean(test_scores)
    print(f"  测试集平均分数: {avg_test:.4f}")
    
    # 6. 计算泛化性差距
    generalization_gap = abs(avg_train - avg_test)
    print(f"\n  泛化性差距: {generalization_gap:.4f}")
    
    # 7. 评判
    if generalization_gap < 0.1:
        verdict = "优秀（差距 < 0.1）"
    elif generalization_gap < 0.3:
        verdict = "良好（差距 < 0.3）"
    elif generalization_gap < 0.5:
        verdict = "一般（差距 < 0.5）"
    else:
        verdict = "需改进（差距 ≥ 0.5）"
    
    print(f"  评判: {verdict}")
    
    # 8. 反向测试
    print("\n" + "="*60)
    print("反向测试: Air Quality → SECOM")
    print("="*60)
    
    perception2 = FiveSensePerception()
    logic_engine2 = LogicEngine()
    emotion_engine2 = EmotionEngine()
    executor2 = Executor()
    scorer2 = QuantizationScorer()
    
    print("\n[训练] 在 Air Quality 数据上训练...")
    train_scores2 = []
    n_train2 = min(100, len(X_train_aq))
    
    for i in range(n_train2):
        sample = X_train_aq[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_train_aq[i])
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
        result = run_one_sample(
            perception2, logic_engine2, emotion_engine2,
            executor2, scorer2, vec, label, modality="olfactory"
        )
        train_scores2.append(result["aggregated_score"])
    
    avg_train2 = np.mean(train_scores2)
    print(f"  训练集平均分数: {avg_train2:.4f}")
    
    print("\n[测试] 在 SECOM 数据上测试（跨领域）...")
    test_scores2 = []
    n_test2 = min(50, len(X_test_secom))
    
    for i in range(n_test2):
        sample = X_test_secom[i]
        sample = np.nan_to_num(sample, nan=0.0)
        label = float(y_test_secom[i])
        vec = sample[:512]
        vec = np.pad(vec, (0, max(0, 512 - len(vec))), mode="constant")[:512]
        result = run_one_sample(
            perception2, logic_engine2, emotion_engine2,
            executor2, scorer2, vec, label, modality="visual"
        )
        test_scores2.append(result["aggregated_score"])
    
    avg_test2 = np.mean(test_scores2)
    print(f"  测试集平均分数: {avg_test2:.4f}")
    
    generalization_gap2 = abs(avg_train2 - avg_test2)
    print(f"\n  泛化性差距（反向）: {generalization_gap2:.4f}")
    
    if generalization_gap2 < 0.1:
        verdict2 = "优秀（差距 < 0.1）"
    elif generalization_gap2 < 0.3:
        verdict2 = "良好（差距 < 0.3）"
    elif generalization_gap2 < 0.5:
        verdict2 = "一般（差距 < 0.5）"
    else:
        verdict2 = "需改进（差距 ≥ 0.5）"
    
    print(f"  评判（反向）: {verdict2}")
    
    # 9. 总结
    print("\n" + "="*60)
    print("跨数据集泛化性测试总结")
    print("="*60)
    print(f"  正向（SECOM → Air Quality）: {generalization_gap:.4f} - {verdict}")
    print(f"  反向（Air Quality → SECOM）: {generalization_gap2:.4f} - {verdict2}")
    print(f"  平均泛化性差距: {(generalization_gap + generalization_gap2)/2:.4f}")
    
    return {
        "gap_forward": generalization_gap,
        "gap_backward": generalization_gap2,
        "verdict_forward": verdict,
        "verdict_backward": verdict2,
    }


if __name__ == "__main__":
    demo_cross_domain_generalization()
