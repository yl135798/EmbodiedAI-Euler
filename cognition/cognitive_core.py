"""
认知层 - 逻辑推理引擎 + 情绪模拟引擎
实现"知"：逻辑思考 + 情绪思考
"""

import numpy as np
from typing import Dict, Tuple, List
import json


class LogicEngine:
    """
    逻辑推理引擎
    基于符号逻辑 + 神经推理的混合架构
    """

    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        # 简化的推理权重矩阵
        self.W_logic = np.random.randn(state_dim, hidden_dim) * 0.1
        self.W_out = np.random.randn(hidden_dim, state_dim) * 0.1
        self.bias = np.zeros(hidden_dim)

    def reason(self, fused_vector: np.ndarray, context: Dict = None) -> Dict:
        """
        对融合感知向量进行逻辑推理
        返回: {
            "conclusion": np.ndarray,   # 推理结论向量
            "confidence": float,         # 推理置信度
            "reasoning_chain": List[str] # 推理链（可解释性）
        }
        """
        # 神经推理层
        hidden = np.tanh(fused_vector @ self.W_logic + self.bias)
        conclusion = np.tanh(hidden @ self.W_out)

        # 逻辑置信度：基于向量范数和一致性
        confidence = float(
            np.clip(np.linalg.norm(conclusion) / (np.linalg.norm(fused_vector) + 1e-8), 0, 1)
        )

        # 简化的推理链（实际可接入LLM生成）
        reasoning_chain = self._extract_reasoning_chain(fused_vector, conclusion)

        return {
            "conclusion": conclusion,
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
        }

    def _extract_reasoning_chain(self, inp: np.ndarray, out: np.ndarray) -> List[str]:
        """提取可解释的推理步骤"""
        return [
            f"感知向量范数: {np.linalg.norm(inp):.3f}",
            f"推理输出范数: {np.linalg.norm(out):.3f}",
            f"活跃维度: {np.sum(np.abs(out) > 0.5)}/{self.state_dim}",
        ]


class EmotionEngine:
    """
    情绪思考引擎
    基于 Plutchik 情绪轮 + 情绪向量空间
    八种基本情绪: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    """

    EMOTIONS = ["joy", "trust", "fear", "surprise",
                 "sadness", "disgust", "anger", "anticipation"]

    def __init__(self, state_dim: int = 512):
        self.state_dim = state_dim
        self.emotion_basis = self._init_emotion_basis()
        self.current_emotion = np.zeros(len(self.EMOTIONS))

    def _init_emotion_basis(self) -> np.ndarray:
        """初始化情绪基向量（每种情绪对应一个方向）"""
        np.random.seed(123)
        basis = np.random.randn(len(self.EMOTIONS), self.state_dim)
        # 归一化
        basis = basis / (np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8)
        return basis

    def infer_emotion(self, fused_vector: np.ndarray) -> Dict:
        """
        从感知向量推断情绪状态
        返回情绪向量 + 主导情绪
        """
        # 计算感知向量与各情绪的相似度
        fused_norm = fused_vector / (np.linalg.norm(fused_vector) + 1e-8)
        similarities = [
            float(np.dot(fused_norm, basis_i))
            for basis_i in self.emotion_basis
        ]

        # Softmax 得到情绪分布
        exp_sim = np.exp(np.array(similarities) * 5)  # 温度系数=5
        emotion_dist = exp_sim / (np.sum(exp_sim) + 1e-8)

        self.current_emotion = emotion_dist

        dominant_idx = int(np.argmax(emotion_dist))
        dominant_emotion = self.EMOTIONS[dominant_idx]

        return {
            "emotion_distribution": dict(zip(self.EMOTIONS, emotion_dist.tolist())),
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": float(emotion_dist[dominant_idx]),
            "valence": float(self._compute_valence(emotion_dist)),
            "arousal": float(np.linalg.norm(emotion_dist)),
        }

    def _compute_valence(self, dist: np.ndarray) -> float:
        """计算情绪效价 [-1, 1]：正值=愉悦，负值=不愉悦"""
        positive = dist[0] + dist[1] + dist[7]  # joy, trust, anticipation
        negative = dist[2] + dist[4] + dist[5] + dist[6]  # fear, sadness, disgust, anger
        return float(positive - negative)

    def blend(self, logic_result: Dict, emotion_result: Dict) -> np.ndarray:
        """
        逻辑与情绪的融合（"知"的核心）
        逻辑主导 + 情绪调制
        """
        logic_vec = logic_result["conclusion"]
        emotion_weight = emotion_result["emotion_intensity"]

        # 情绪效价调制逻辑推理的强度
        valence = emotion_result["valence"]
        modulation = 1.0 + 0.3 * valence  # 积极情绪增强推理，消极情绪减弱

        blended = logic_vec * modulation * (1 - 0.2 * emotion_weight)
        # 情绪较强时适当降低逻辑权重，模拟"情绪影响判断"

        return blended
