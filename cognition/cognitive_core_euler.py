# -*- coding: utf-8 -*-
"""
认知层 - 欧拉恒等式同构架构
核心公式: e^(iθ) = cos(θ) + i·sin(θ)

知 (Knowledge)  = 复数 z = r · e^(iθ)
  - 实部 (Real)   = Logic  (符号推理, cos(θ))
  - 虚部 (Imag)   = Emotion (情绪空间, sin(θ))
  - 模长 (|z|)    = 置信度
  - 幅角 (arg(z)) = 认知状态

行 (Action)    = 复平面上的向量 (r, θ)
  - 钱 (Resource) = |z| (模长, 资源分配)
  - 权 (Power)    = θ (幅角, 权力分配)

量化 (Score)   = f(|z|, θ)
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple
import json


class EulerCognitiveEngine:
    """
    欧拉同构认知引擎
    使用复平面表示认知状态

    核心公式:
        z = r · e^(iθ)
        e^(iθ) = cos(θ) + i·sin(θ)

    映射:
        Logic (符号推理)   → 实部 (cos(θ))
        Emotion (情绪空间)   → 虚部 (sin(θ))
        融合状态            → 复数 z
    """

    def __init__(self, state_dim: int = 512, n_emotions: int = 8):
        """
        初始化欧拉认知引擎

        Args:
            state_dim: 状态维度
            n_emotions: 情绪维度（Plutchik 八维）
        """
        self.state_dim = state_dim
        self.n_emotions = n_emotions

        # 逻辑权重矩阵（实部变换）
        self.W_logic_real = np.random.randn(state_dim, state_dim) * 0.1

        # 情绪权重矩阵（虚部变换）
        self.W_emotion_imag = np.random.randn(state_dim, n_emotions) * 0.1

        # 情绪基向量（8 种基本情绪对应单位圆上的角度）
        # Plutchik 八维情绪空间映射 to 复平面
        self.emotion_angles = np.linspace(0, 2*np.pi, n_emotions, endpoint=False)
        # emotions: joy, trust, fear, surprise, sadness, disgust, anger, anticipation

        # 当前认知状态（复数向量）
        self.current_state = np.zeros(state_dim, dtype=complex)

        # 历史轨迹（用于计算 φ 和 e）
        self.trajectory = []  # 存储历史复数状态

    def reason(self, fused_vector: np.ndarray) -> Dict:
        """
        对融合感知向量进行逻辑推理（实部计算）

        返回:
            {
                "logic_vector": np.ndarray,  # 逻辑推理向量（实部）
                "confidence": float,           # 推理置信度（模长）
                "reasoning_chain": List[str],  # 推理链
            }
        """
        # 1. 逻辑推理（实部变换）
        logic_real = np.tanh(fused_vector @ self.W_logic_real)  # 实部

        # 2. 计算置信度（归一化范数，sigmoid 压缩到 [0, 1]）
        norm = np.linalg.norm(logic_real)
        # sigmoid: 将范数映射到 [0, 1]
        confidence = float(1.0 / (1.0 + np.exp(-norm)))
        # 或者直接用归一化：confidence = float(norm / (np.sqrt(self.state_dim) + 1e-8))
        confidence = np.clip(confidence, 0.0, 1.0)

        # 3. 提取推理链（可解释性）
        reasoning_chain = self._extract_reasoning_chain(fused_vector, logic_real)

        return {
            "logic_vector": logic_real,  # 实部
            "confidence": confidence,
            "reasoning_chain": reasoning_chain,
        }

    def infer_emotion(self, fused_vector: np.ndarray) -> Dict:
        """
        推断情绪状态（虚部计算）

        返回:
            {
                "emotion_vector": np.ndarray,  # 情绪向量（虚部）
                "emotion_distribution": Dict,    # 情绪分布（8 维）
                "emotion_angle": float,          # 情绪角度 θ（弧度）
            }
        """
        # 1. 情绪推断（虚部变换）
        emotion_logits = fused_vector @ self.W_emotion_imag  # (n_emotions,)

        # ✅ 修复：数值稳定 Softmax（防止 exp() 溢出）
        logits_max = np.max(emotion_logits)
        emotion_logits_stable = emotion_logits - logits_max  # 减最大值
        emotion_probs = np.exp(emotion_logits_stable) / (np.sum(np.exp(emotion_logits_stable)) + 1e-8)

        # 2. 计算情绪角度（加权平均）
        emotion_angle = np.sum(emotion_probs * self.emotion_angles)  # 期望角度

        # 3. 构造虚部向量（正弦变换）
        emotion_imag = np.sin(emotion_angle) * np.ones(self.state_dim)  # 虚部（简化）

        # 4. 情绪分布（Plutchik 八维）
        emotion_names = ["joy", "trust", "fear", "surprise",
                        "sadness", "disgust", "anger", "anticipation"]
        emotion_distribution = {
            name: float(prob) for name, prob in zip(emotion_names, emotion_probs)
        }

        return {
            "emotion_vector": emotion_imag,  # 虚部
            "emotion_distribution": emotion_distribution,
            "emotion_angle": float(emotion_angle),
        }

    def blend(self, logic_result: Dict, emotion_result: Dict) -> complex:
        """
        融合逻辑与情绪（欧拉公式）

        核心公式:
            z = r · e^(iθ)
            e^(iθ) = cos(θ) + i·sin(θ)

        其中:
            r = confidence (逻辑置信度) → 模长
            θ = emotion_angle (情绪角度) → 幅角

        返回:
            z: 融合后的复数认知状态
        """
        # 1. 提取模长 r 和幅角 θ
        r = logic_result["confidence"]  # 模长 = 逻辑置信度
        theta = emotion_result["emotion_angle"]  # 幅角 = 情绪角度

        # 2. 欧拉公式: z = r · e^(iθ) = r · (cos(θ) + i·sin(θ))
        z = r * cmath.exp(1j * theta)

        # 3. 更新当前状态（存储到轨迹）
        self.current_state = z
        self.trajectory.append(z)

        return z

    def compute_action(self, z: complex) -> Tuple[float, float]:
        """
        从复数认知状态计算行动向量

        行 (Action) = 复平面上的向量 (r, θ)
          - 钱 (Resource) = |z| (模长, 资源分配)
          - 权 (Power)    = θ (幅角, 权力分配)

        返回:
            (resource, power): 资源和权力分配
        """
        # 1. 模长 = 资源分配（钱）
        resource = np.abs(z)  # |z|

        # 2. 幅角 = 权力分配（权）
        power = np.angle(z)  # arg(z), 范围 [-π, π]

        # 3. 归一化到 [0, 1]
        resource_norm = np.clip(resource, 0.0, 1.0)
        power_norm = (power + np.pi) / (2 * np.pi)  # 映射到 [0, 1]

        return float(resource_norm), float(power_norm)

    def score(self, z: complex) -> Dict:
        """
        量化评分（基于复数状态）

        量化 = f(|z|, θ)
          - 主评分 = |z| (模长)
          - 辅助评分 = cos(θ) (实部投影)
          - 情绪评分 = sin(θ) (虚部投影)

        返回:
            {
                "aggregated_score": float,  # 综合评分
                "magnitude": float,         # 模长 |z|
                "phase": float,             # 幅角 θ
                "real_part": float,         # 实部 (cos(θ))
                "imag_part": float,         # 虚部 (sin(θ))
            }
        """
        # 1. 计算模长和幅角
        magnitude = np.abs(z)
        phase = np.angle(z)

        # 2. 计算实部和虚部
        real_part = z.real
        imag_part = z.imag

        # 3. 综合评分（加权平均）
        #    主评分 = 模长（重要性最高）
        #    辅助评分 = 实部（逻辑一致性）
        #    情绪评分 = 虚部（情绪强度）
        aggregated_score = 0.6 * magnitude + 0.3 * abs(real_part) + 0.1 * abs(imag_part)
        aggregated_score = np.clip(aggregated_score, 0.0, 1.0)

        return {
            "aggregated_score": float(aggregated_score),
            "magnitude": float(magnitude),
            "phase": float(phase),
            "real_part": float(real_part),
            "imag_part": float(imag_part),
        }

    def _extract_reasoning_chain(self, inp: np.ndarray, logic: np.ndarray) -> List[str]:
        """提取可解释的推理步骤"""
        # 计算逻辑向量的范数（标量）
        logic_norm = float(np.linalg.norm(logic))
        input_norm = float(np.linalg.norm(inp))
        
        # 获取前两个元素构造复数（用于展示）
        logic_complex = complex(float(logic[0]), float(logic[1])) if len(logic) >= 2 else complex(0, 0)
        angle = np.angle(logic_complex)
        
        return [
            f"感知向量范数: {input_norm:.3f}",
            f"逻辑推理范数: {logic_norm:.3f}",
            f"活跃维度: {np.sum(np.abs(logic) > 0.5)}/{self.state_dim}",
            f"欧拉形式（示例）: z = {logic_norm:.3f} · e^(i·{angle:.3f})",
        ]

    def get_trajectory(self) -> List[complex]:
        """获取认知轨迹（用于可视化）"""
        return self.trajectory

    def reset(self):
        """重置认知状态"""
        self.current_state = np.zeros(self.state_dim, dtype=complex)
        self.trajectory = []


# -*- 为了向后兼容，保留原有接口 -*-
class LogicEngine(EulerCognitiveEngine):
    """向后兼容：逻辑推理引擎（使用欧拉同构架构）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reason(self, fused_vector: np.ndarray, context: Dict = None) -> Dict:
        """兼容原有接口"""
        result = super().reason(fused_vector)
        return {
            "conclusion": result["logic_vector"],
            "confidence": result["confidence"],
            "reasoning_chain": result["reasoning_chain"],
        }


class EmotionEngine(EulerCognitiveEngine):
    """向后兼容：情绪思考引擎（使用欧拉同构架构）"""
    EMOTIONS = ["joy", "trust", "fear", "surprise",
                 "sadness", "disgust", "anger", "anticipation"]

    def __init__(self, state_dim: int = 512):
        super().__init__(state_dim=state_dim, n_emotions=len(self.EMOTIONS))

    def infer_emotion(self, fused_vector: np.ndarray) -> Dict:
        """兼容原有接口"""
        return super().infer_emotion(fused_vector)

    def blend(self, logic_result: Dict, emotion_result: Dict) -> Dict:
        """兼容原有接口：返回字典而非复数"""
        z = super().blend(logic_result, emotion_result)
        return {
            "blended_vector": np.array([z.real, z.imag]),  # 兼容旧接口
            "complex_state": z,
            "magnitude": np.abs(z),
            "phase": np.angle(z),
        }

    def get_emotion_distribution(self) -> Dict:
        """获取当前情绪分布"""
        return self.infer_emotion(np.zeros(self.state_dim))["emotion_distribution"]
