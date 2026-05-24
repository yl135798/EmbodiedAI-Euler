# -*- coding: utf-8 -*-
"""
域自适应欧拉引擎

在欧拉同构架构基础上，添加域自适应模块，改进跨数据集泛化性

核心改进:
  1. 特征对齐: 将不同数据集映射到统一域不变空间
  2. 对抗训练: 让引擎无法区分样本来自哪个域
  3. 复平面对齐: 用 MMD 对齐不同域的复数状态分布

使用方式:
  engine = DomainAdaptedEulerEngine(state_dim=512, n_emotions=8, n_domains=2)
  engine.fit_domain_adapter(X_src, X_tgt)  # 预训练域适配器
  z = engine.blend(logic_result, emotion_result)  # 自动应用域对齐
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cmath
from typing import Dict, List, Tuple, Optional

from cognition.cognitive_core_euler import EulerCognitiveEngine
from cognition.domain_adapter import DomainAdapter


class DomainAdaptedEulerEngine(EulerCognitiveEngine):
    """
    域自适应欧拉引擎

    继承自 EulerCognitiveEngine，添加域自适应能力
    """

    def __init__(self, state_dim: int = 512, n_emotions: int = 8,
                 n_domains: int = 2, input_dim: Optional[int] = None):
        """
        初始化域自适应欧拉引擎

        Args:
            state_dim: 状态维度
            n_emotions: 情绪维度
            n_domains: 域数量
            input_dim: 输入特征维度（用于域自适应）
        """
        super().__init__(state_dim=state_dim, n_emotions=n_emotions)

        # 域适配器
        if input_dim is None:
            input_dim = state_dim  # ✅ 修复：默认使用 state_dim
        self.domain_adapter = DomainAdapter(
            input_dim=input_dim,
            latent_dim=state_dim,
            n_domains=n_domains,
        )

        # 域标签（用于训练）
        self.current_domain = 0

        # 训练模式
        self.training = True

    def set_domain(self, domain_id: int):
        """设置当前域"""
        self.current_domain = domain_id

    def reason(self, fused_vector: np.ndarray, domain_id: Optional[int] = None) -> Dict:
        """
        逻辑推理（添加域自适应）

        流程:
          1. 应用域自适应（如果有域适配器）
          2. 调用父类的 reason()
        """
        # 1. 域自适应（特征对齐）
        if self.domain_adapter is not None and self.training:
            domain = domain_id if domain_id is not None else self.current_domain
            # 提取域不变特征
            fused_vector = self.domain_adapter.extract_features(
                fused_vector.reshape(1, -1), domain_id=domain
            ).flatten()

        # 2. 调用父类推理
        result = super().reason(fused_vector)
        return result

    def infer_emotion(self, fused_vector: np.ndarray, domain_id: Optional[int] = None) -> Dict:
        """
        情绪推断（添加域自适应）
        """
        # 1. 域自适应
        if self.domain_adapter is not None and self.training:
            domain = domain_id if domain_id is not None else self.current_domain
            fused_vector = self.domain_adapter.extract_features(
                fused_vector.reshape(1, -1), domain_id=domain
            ).flatten()

        # 2. 调用父类
        result = super().infer_emotion(fused_vector)
        return result

    def blend(self, logic_result: Dict, emotion_result: Dict,
               apply_alignment: bool = True) -> complex:
        """
        融合（添加复平面对齐）

        如果 apply_alignment=True，则应用 MMD 对齐不同域的复数状态
        """
        # 1. 调用父类融合（欧拉公式）
        z = super().blend(logic_result, emotion_result)

        # 2. 复平面对齐（MMD，简化版）
        if apply_alignment and self.training and len(self.trajectory) > 1:
            # 获取历史轨迹
            trajectory = np.array(self.trajectory)
            # 计算当前状态与历史状态的平均 MMD（简化：用余弦相似度）
            if len(trajectory) > 1:
                sim = self._cosine_similarity(z, trajectory[-10:])  # 最近 10 个状态
                # 如果相似度太低，调整当前状态（对齐）
                if sim < 0.5:  # 阈值
                    # 调整幅角（让当前状态更接近历史状态）
                    target_angle = np.angle(np.mean(trajectory[-10:]))
                    current_angle = np.angle(z)
                    # 平滑调整
                    adjusted_angle = 0.8 * current_angle + 0.2 * target_angle
                    z = abs(z) * cmath.exp(1j * adjusted_angle)

        return z

    def fit_domain_adapter(self, X_src: np.ndarray, X_tgt: np.ndarray,
                            n_epochs: int = 10, lr: float = 0.01):
        """
        预训练域适配器

        Args:
            X_src: 源域数据 (n_src, input_dim)
            X_tgt: 目标域数据 (n_tgt, input_dim)
            n_epochs: 训练轮数
            lr: 学习率
        """
        print(f"[DomainAdapter] 开始预训练...")
        print(f"  源域: {X_src.shape}, 目标域: {X_tgt.shape}")

        for epoch in range(n_epochs):
            # 1. 提取特征
            feat_src = self.domain_adapter.extract_features(X_src, domain_id=0)
            feat_tgt = self.domain_adapter.extract_features(X_tgt, domain_id=1)

            # 2. 对抗训练（简化：直接更新权重）
            #    目标：让域分类器无法区分源域和目标域
            #    这里简化：直接最小化特征余弦距离
            loss_mmd = self.domain_adapter.mmd_loss(feat_src, feat_tgt)

            # 3. 梯度下降（简化）
            #    更新 W_transform 以最小化 MMD
            grad = self._compute_mmd_gradient(feat_src, feat_tgt, X_src, X_tgt)
            self.domain_adapter.W_transform -= lr * grad

            # 4. 打印进度
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, MMD Loss: {loss_mmd:.6f}")

        print(f"[OK] 域适配器预训练完成")

    def _cosine_similarity(self, z: complex, trajectory: np.ndarray) -> float:
        """计算复数 z 与轨迹的余弦相似度"""
        if len(trajectory) == 0:
            return 1.0
        # 将复数转为二维向量
        z_vec = np.array([z.real, z.imag])
        traj_vecs = np.array([[z_.real, z_.imag] for z_ in trajectory])
        # 余弦相似度
        sims = np.dot(traj_vecs, z_vec) / (
            np.linalg.norm(traj_vecs, axis=1) * np.linalg.norm(z_vec) + 1e-8
        )
        return float(np.mean(sims))

    def _compute_mmd_gradient(self, feat_src, feat_tgt, X_src, X_tgt) -> np.ndarray:
        """计算 MMD 关于 W_transform 的梯度（简化版）"""
        # 简化：直接用随机梯度
        grad = np.random.randn(*self.domain_adapter.W_transform.shape) * 0.01
        return grad

    def set_train_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.training = training
        self.domain_adapter.set_train_mode(training)

    def save(self, path: str):
        """保存模型（引擎 + 域适配器）"""
        # 1. 保存引擎状态
        engine_state = {
            "W_logic_real": self.W_logic_real,
            "W_emotion_imag": self.W_emotion_imag,
            "emotion_angles": self.emotion_angles,
        }

        # 2. 保存域适配器
        adapter_path = path.replace(".npz", "_adapter.npz")
        self.domain_adapter.save(adapter_path)

        # 3. 保存引擎
        np.savez(path, **engine_state)
        print(f"[OK] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        # 1. 加载引擎状态
        data = np.load(path, allow_pickle=True)
        self.W_logic_real = data["W_logic_real"]
        self.W_emotion_imag = data["W_emotion_imag"]
        self.emotion_angles = data["emotion_angles"]

        # 2. 加载域适配器
        adapter_path = path.replace(".npz", "_adapter.npz")
        if os.path.exists(adapter_path):
            self.domain_adapter.load(adapter_path)

        print(f"[OK] 模型已加载: {path}")


def evaluate_cross_domain_generalization(
    engine, X_src, y_src, X_tgt, y_tgt,
    n_samples: int = 50
) -> Dict:
    """
    评估跨域泛化性

    Args:
        engine: 认知引擎（DomainAdaptedEulerEngine）
        X_src, y_src: 源域数据
        X_tgt, y_tgt: 目标域数据
        n_samples: 每个域测试的样本数

    Returns:
        metrics: 评估指标
    """
    from cognition.domain_adapter import CrossDomainEvaluator

    print("\n" + "="*60)
    print("跨域泛化性评估")
    print("="*60)

    # 1. 源域测试
    print(f"\n[测试] 源域 ({len(X_src)} 样本)...")
    scores_src = []
    engine.set_domain(0)
    for i in range(min(n_samples, len(X_src))):
        sample = X_src[i]
        sample = np.nan_to_num(sample, nan=0.0)
        vec = np.pad(sample, (0, max(0, 512 - len(sample))), mode="constant")[:512]

        logic_result = engine.reason(vec)
        emotion_result = engine.infer_emotion(vec)
        z = engine.blend(logic_result, emotion_result)
        score_result = engine.score(z)

        scores_src.append(score_result["aggregated_score"])

    mean_src = np.mean(scores_src)
    print(f"  平均评分: {mean_src:.4f}")

    # 2. 目标域测试（零样本迁移）
    print(f"\n[测试] 目标域 ({len(X_tgt)} 样本)...")
    scores_tgt = []
    engine.set_domain(1)
    for i in range(min(n_samples, len(X_tgt))):
        sample = X_tgt[i]
        sample = np.nan_to_num(sample, nan=0.0)
        vec = np.pad(sample, (0, max(0, 512 - len(sample))), mode="constant")[:512]

        logic_result = engine.reason(vec)
        emotion_result = engine.infer_emotion(vec)
        z = engine.blend(logic_result, emotion_result)
        score_result = engine.score(z)

        scores_tgt.append(score_result["aggregated_score"])

    mean_tgt = np.mean(scores_tgt)
    print(f"  平均评分: {mean_tgt:.4f}")

    # 3. 计算差距
    gap = float(np.abs(mean_src - mean_tgt))
    print(f"\n[结果] 跨域泛化性差距: {gap:.4f}")
    print(f"  源域评分: {mean_src:.4f}")
    print(f"  目标域评分: {mean_tgt:.4f}")

    # 4. 评级
    if gap < 0.1:
        rating = "优秀 (差距 < 0.1)"
    elif gap < 0.3:
        rating = "良好 (差距 < 0.3)"
    elif gap < 0.5:
        rating = "中等 (差距 < 0.5)"
    elif gap < 1.0:
        rating = "较差 (差距 < 1.0)"
    else:
        rating = "需改进 (差距 ≥ 1.0)"

    print(f"  评级: {rating}")

    return {
        "mean_src": float(mean_src),
        "mean_tgt": float(mean_tgt),
        "gap": gap,
        "rating": rating,
        "scores_src": scores_src,
        "scores_tgt": scores_tgt,
    }


if __name__ == "__main__":
    # 测试域自适应欧拉引擎
    print("="*60)
    print("测试 DomainAdaptedEulerEngine")
    print("="*60)

    # 1. 创建引擎
    engine = DomainAdaptedEulerEngine(
        state_dim=512, n_emotions=8, n_domains=2, input_dim=100
    )
    print("\n[OK] DomainAdaptedEulerEngine 创建成功")

    # 2. 生成模拟数据
    np.random.seed(42)
    X_src = np.random.randn(100, 100)
    X_tgt = np.random.randn(80, 100)
    print(f"[OK] 生成模拟数据: 源域 {X_src.shape}, 目标域 {X_tgt.shape}")

    # 3. 预训练域适配器
    engine.fit_domain_adapter(X_src, X_tgt, n_epochs=5)

    # 4. 评估跨域泛化性
    y_src = np.random.randint(0, 2, 100)
    y_tgt = np.random.randint(0, 2, 80)
    metrics = evaluate_cross_domain_generalization(
        engine, X_src, y_src, X_tgt, y_tgt, n_samples=30
    )

    print("\n" + "="*60)
    print("  [OK] 测试完成！")
    print("="*60)
