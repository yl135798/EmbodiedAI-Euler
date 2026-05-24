"""
量化输出层 - 评分系统 + 结果报告
将"知行合一"的结果量化为可解释的指标
"""

import numpy as np
from typing import Dict, List
import json
import time


class QuantizationScorer:
    """
    量化评分系统
    将执行结果转化为可比较、可追踪的量化指标
    """

    def __init__(self):
        self.score_history: List[Dict] = []
        self.metrics = [
            "cognition_quality",    # 认知质量
            "resource_efficiency",  # 资源效率
            "power_fairness",       # 权力公平性（基尼系数）
            "execution_confidence", # 执行置信度
            "generalization_gap",   # 泛化性差距（训练vs测试）
        ]

    def score(self, execution_result: Dict, ground_truth: float = None) -> Dict:
        """
        对一次执行结果进行多维评分
        Returns:
            score_vector: 各维度分数
            aggregated: 综合分数
            diagnostics: 诊断信息
        """
        qr = execution_result
        quantized = qr.get("quantized_score", 0.0)
        power_weights = qr.get("power_weights", {})
        resource_factor = qr.get("resource_factor", 1.0)

        # 1. 认知质量
        cognition_quality = min(quantized, 1.0)

        # 2. 资源效率
        resource_efficiency = resource_factor

        # 3. 权力公平性（基尼系数，越低越公平）
        weights = np.array(list(power_weights.values())) if power_weights else np.array([1.0])
        gini = self._gini_coefficient(weights)
        power_fairness = 1.0 - gini

        # 4. 执行置信度
        output_vec = np.array(qr.get("output_vector", []))
        execution_confidence = float(np.mean(np.abs(output_vec))) if len(output_vec) > 0 else 0.0

        # 5. 泛化性差距（需外部提供 ground_truth）
        generalization_gap = 0.0
        if ground_truth is not None:
            generalization_gap = abs(quantized - ground_truth)

        score_vector = {
            "cognition_quality": float(cognition_quality),
            "resource_efficiency": float(resource_efficiency),
            "power_fairness": float(power_fairness),
            "execution_confidence": float(execution_confidence),
            "generalization_gap": float(generalization_gap),
        }

        # 综合分数（加权平均）
        weights = {
            "cognition_quality": 0.3,
            "resource_efficiency": 0.2,
            "power_fairness": 0.2,
            "execution_confidence": 0.2,
            "generalization_gap": 0.1,
        }
        aggregated = sum(
            score_vector[k] * w for k, w in weights.items()
        ) / sum(weights.values())

        diagnostics = self._diagnose(score_vector, qr)

        result = {
            "score_vector": score_vector,
            "aggregated_score": float(aggregated),
            "diagnostics": diagnostics,
            "timestamp": time.time(),
        }

        self.score_history.append(result)
        return result

    def _gini_coefficient(self, x: np.ndarray) -> float:
        """计算基尼系数"""
        if len(x) == 0 or np.sum(x) == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n
        return float(gini)

    def _diagnose(self, scores: Dict, qr: Dict) -> List[str]:
        diagnostics = []
        if scores["cognition_quality"] < 0.3:
            diagnostics.append("⚠️ 认知质量低：感知数据不足或融合失败")
        if scores["resource_efficiency"] < 0.5:
            diagnostics.append("⚠️ 资源效率偏低：预算不足或分配不合理")
        if scores["power_fairness"] < 0.4:
            diagnostics.append("⚠️ 权力过度集中：少数agent控制决策")
        if scores["execution_confidence"] < 0.3:
            diagnostics.append("⚠️ 执行置信度低：输出向量不稳定")
        if not diagnostics:
            diagnostics.append("✅ 各项指标正常")
        return diagnostics

    def generalization_test(self,
                          train_scores: List[float],
                          test_scores: List[float]) -> Dict:
        """
        泛化性测试：比较训练集和测试集的表现差距
        这是体现通用性和泛化性的核心指标
        """
        train_mean = np.mean(train_scores)
        test_mean = np.mean(test_scores)
        gap = abs(train_mean - test_mean)
        # 取相同长度比较
        n = min(len(train_scores), len(test_scores))
        std_gap = float(np.std(np.array(train_scores[:n]) - np.array(test_scores[:n])))

        train_mean = float(np.mean(train_scores[:n]))
        test_mean = float(np.mean(test_scores[:n]))
        gap = abs(train_mean - test_mean)

        # 泛化性得分：差距越小越好
        generalization_score = max(0.0, 1.0 - gap)

        return {
            "train_mean": float(train_mean),
            "test_mean": float(test_mean),
            "generalization_gap": float(gap),
            "generalization_std": float(std_gap),
            "generalization_score": float(generalization_score),
            "verdict": "泛化性好" if generalization_score > 0.7 else "泛化性需改进",
        }
