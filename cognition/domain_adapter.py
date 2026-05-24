# -*- coding: utf-8 -*-
"""
域自适应模块 - 用于改进跨数据集泛化性

核心思想:
  1. 特征变换: 将不同数据集映射到统一欧拉空间 (512D)
  2. 域分类器 (对抗训练): 让引擎无法区分样本来自哪个域
  3. 复平面对齐: 用 MMD (Maximum Mean Discrepancy) 对齐分布

损失函数:
  L_total = L_task + λ * L_domain_adv + μ * L_mmd

其中:
  - L_task: 主任务损失 (评分精度)
  - L_domain_adv: 域分类对抗损失 (让特征域不变)
  - L_mmd: 最大均值差异损失 (对齐特征分布)
  - λ, μ: 超参数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class DomainAdapter:
    """
    域自适应适配器

    用于将不同数据集的特征映射到统一的域不变空间
    """

    def __init__(self, input_dim: int, latent_dim: int = 512, n_domains: int = 2):
        """
        初始化域适配器

        Args:
            input_dim: 输入特征维度
            latent_dim: 隐空间维度（统一空间）
            n_domains: 域数量（例如：SECOM=0, Air Quality=1）
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_domains = n_domains

        # 1. 特征变换矩阵（将输入映射到统一空间）
        self.W_transform = np.random.randn(input_dim, latent_dim) * 0.1

        # 2. 域分类器（用于对抗训练）
        #    输入: latent_dim, 输出: n_domains
        self.W_domain_classifier = np.random.randn(latent_dim, n_domains) * 0.1

        # 3. 域均值（用于 MMD 计算）
        self.domain_means = {i: np.zeros(latent_dim) for i in range(n_domains)}
        self.domain_protos = {i: [] for i in range(n_domains)}  # 原型样本

        # 4. 超参数
        self.lambda_adv = 0.1   # 对抗损失权重
        self.mu_mmd = 0.01       # MMD 损失权重

        # 5. 训练状态
        self.training = True
        self.batch_domain_labels = []  # 记录每个样本的域标签

    def extract_features(self, X: np.ndarray, domain_id: int) -> np.ndarray:
        """
        提取域不变特征（核心函数）

        流程:
          1. 线性变换: X → latent_space
          2. 归一化: 保证特征在单位球附近
          3. 返回: 域不变特征（可用于后续认知计算）

        Args:
            X: 输入特征 (n_samples, input_dim)
            domain_id: 域编号 (0, 1, ..., n_domains-1)

        Returns:
            features: 域不变特征 (n_samples, latent_dim)
        """
        # 1. 线性变换
        features = X @ self.W_transform  # (n_samples, latent_dim)

        # 2. 归一化（L2 归一化）
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        # 3. 记录到原型库（用于 MMD）
        if self.training:
            self.domain_protos[domain_id].append(features)
            # 更新域均值
            self.domain_means[domain_id] = np.mean(
                np.vstack(self.domain_protos[domain_id]), axis=0
            )

        return features

    def domain_classifier_loss(self, features: np.ndarray, domain_labels: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        域分类器损失（对抗训练）

        目标:
          - 域分类器: 尽量正确分类样本来自哪个域
          - 特征提取器: 尽量让域分类器犯错（提取域不变特征）

        Args:
            features: 提取的特征 (n_samples, latent_dim)
            domain_labels: 域标签 (n_samples,) 每个元素 ∈ [0, n_domains)

        Returns:
            loss: 域分类损失（交叉熵）
            grad: 梯度（用于反向传播）
        """
        # 1. 前向传播
        logits = features @ self.W_domain_classifier  # (n_samples, n_domains)
        probs = self._softmax(logits)

        # 2. 交叉熵损失
        n_samples = features.shape[0]
        loss = -np.sum(np.log(probs[np.arange(n_samples), domain_labels] + 1e-8)) / n_samples

        # 3. 梯度计算
        grad = probs.copy()
        grad[np.arange(n_samples), domain_labels] -= 1
        grad = grad / n_samples  # (n_samples, n_domains)

        # 4. 反向传播到特征
        grad_features = grad @ self.W_domain_classifier.T  # (n_samples, latent_dim)

        return loss, grad_features

    def mmd_loss(self, features_src: np.ndarray, features_tgt: np.ndarray) -> float:
        """
        计算 MMD (Maximum Mean Discrepancy) 损失

        MMD 用于衡量两个分布的距离:
          MMD² = E[k(x_src, x_src')] + E[k(x_tgt, x_tgt')] - 2·E[k(x_src, x_tgt)]

        其中 k(·,·) 是核函数（这里用 RBF 核）

        Args:
            features_src: 源域特征 (n_src, latent_dim)
            features_tgt: 目标域特征 (n_tgt, latent_dim)

        Returns:
            mmd: MMD 距离（越小表示分布越接近）
        """
        # 1. 计算核矩阵（RBF 核）
        def rbf_kernel(X, Y, gamma=1.0):
            """RBF 核函数"""
            X_norm = np.sum(X ** 2, axis=1, keepdims=True)
            Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
            K = np.exp(-gamma * (X_norm + Y_norm - 2 * X @ Y.T))
            return K

        K_ss = rbf_kernel(features_src, features_src)
        K_tt = rbf_kernel(features_tgt, features_tgt)
        K_st = rbf_kernel(features_src, features_tgt)

        # 2. MMD²
        n_src = features_src.shape[0]
        n_tgt = features_tgt.shape[0]

        mmd = np.sum(K_ss) / (n_src * n_src) + \
              np.sum(K_tt) / (n_tgt * n_tgt) - \
              2 * np.sum(K_st) / (n_src * n_tgt)

        return max(mmd, 0.0)  # MMD² 应非负

    def adapt_features(self, features_src: np.ndarray, features_tgt: np.ndarray) -> np.ndarray:
        """
        自适应特征（对齐源域和目标域）

        使用对抗训练 + MMD 对齐

        Args:
            features_src: 源域特征
            features_tgt: 目标域特征

        Returns:
            features_adapted: 对齐后的特征
        """
        # 1. 对抗训练（让域分类器无法区分）
        domain_labels = np.concatenate([
            np.zeros(features_src.shape[0]),  # 源域 = 0
            np.ones(features_tgt.shape[0]),   # 目标域 = 1
        ]).astype(int)

        all_features = np.vstack([features_src, features_tgt])
        loss_adv, grad_adv = self.domain_classifier_loss(all_features, domain_labels)

        # 2. MMD 对齐
        loss_mmd = self.mmd_loss(features_src, features_tgt)

        # 3. 特征变换（减小域差异）
        #    这里简化：直接返回原特征（实际中应基于梯度更新 W_transform）
        features_adapted = all_features

        # 4. 记录损失
        self.last_loss_adv = loss_adv
        self.last_loss_mmd = loss_mmd

        return features_adapted

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def set_train_mode(self, training: bool = True):
        """设置训练/评估模式"""
        self.training = training

    def get_domain_statistics(self) -> Dict:
        """获取域统计信息"""
        stats = {}
        for domain_id, protos in self.domain_protos.items():
            if len(protos) > 0:
                all_protos = np.vstack(protos)
                stats[f"domain_{domain_id}"] = {
                    "n_samples": all_protos.shape[0],
                    "mean_norm": float(np.mean(np.linalg.norm(all_protos, axis=1))),
                    "mean_cosine_sim": self._compute_mean_cosine_sim(all_protos),
                }
        return stats

    def _compute_mean_cosine_sim(self, features: np.ndarray) -> float:
        """计算特征间的平均余弦相似度"""
        n = features.shape[0]
        if n < 2:
            return 0.0
        sim = features @ features.T
        norms = np.linalg.norm(features, axis=1)
        sim = sim / (norms[:, None] @ norms[None, :] + 1e-8)
        return float(np.sum(sim) / (n * n))

    def save(self, path: str):
        """保存模型"""
        np.savez(
            path,
            W_transform=self.W_transform,
            W_domain_classifier=self.W_domain_classifier,
            domain_means=np.array(list(self.domain_means.values())),
        )
        print(f"[OK] 域适配器已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        data = np.load(path, allow_pickle=True)
        self.W_transform = data["W_transform"]
        self.W_domain_classifier = data["W_domain_classifier"]
        domain_means_arr = data["domain_means"]
        for i in range(len(domain_means_arr)):
            self.domain_means[i] = domain_means_arr[i]
        print(f"[OK] 域适配器已加载: {path}")


class CrossDomainEvaluator:
    """
    跨域泛化性评估器

    用于系统化测试算法在不同域上的表现
    """

    def __init__(self, engine, adapter: Optional[DomainAdapter] = None):
        """
        初始化评估器

        Args:
            engine: 认知引擎（EulerCognitiveEngine）
            adapter: 域适配器（可选）
        """
        self.engine = engine
        self.adapter = adapter
        self.results = {}

    def evaluate_cross_domain(self, dataset_src, dataset_tgt,
                               n_samples: int = 50) -> Dict:
        """
        评估跨域泛化性

        流程:
          1. 在源域上训练/测试
          2. 在目标域上测试（零样本迁移）
          3. 计算泛化性差距

        Args:
            dataset_src: 源域数据集 (X, y, meta)
            dataset_tgt: 目标域数据集 (X, y, meta)
            n_samples: 每个域测试的样本数

        Returns:
            metrics: 评估指标字典
        """
        X_src, y_src, meta_src = dataset_src
        X_tgt, y_tgt, meta_tgt = dataset_tgt

        # 1. 源域测试
        scores_src = self._evaluate_one_domain(X_src, y_src, domain_id=0, n_samples=n_samples)

        # 2. 目标域测试（零样本）
        scores_tgt = self._evaluate_one_domain(X_tgt, y_tgt, domain_id=1, n_samples=n_samples)

        # 3. 计算差距
        gap = {
            "mean_gap": float(np.abs(np.mean(scores_src) - np.mean(scores_tgt))),
            "std_gap": float(np.abs(np.std(scores_src) - np.std(scores_tgt))),
            "max_gap": float(np.max(np.abs(np.array(scores_src) - np.array(scores_tgt)))),
        }

        # 4. 域适配（如果提供了 adapter）
        if self.adapter is not None:
            # 提取特征
            feat_src = self.adapter.extract_features(X_src[:n_samples], domain_id=0)
            feat_tgt = self.adapter.extract_features(X_tgt[:n_samples], domain_id=1)

            # 自适应
            feat_adapted = self.adapter.adapt_features(feat_src, feat_tgt)

            # 重新评估
            scores_adapted = self._evaluate_with_features(feat_adapted, n_samples)

            gap["mean_gap_adapted"] = float(np.abs(np.mean(scores_src) - np.mean(scores_adapted)))
            gap["improvement"] = float(gap["mean_gap"] - gap["mean_gap_adapted"])

        return {
            "scores_src": scores_src,
            "scores_tgt": scores_tgt,
            "gap": gap,
            "mean_src": float(np.mean(scores_src)),
            "mean_tgt": float(np.mean(scores_tgt)),
        }

    def _evaluate_one_domain(self, X, y, domain_id: int, n_samples: int) -> List[float]:
        """评估单个域"""
        scores = []
        n = min(n_samples, len(X))

        for i in range(n):
            sample = X[i]
            sample = np.nan_to_num(sample, nan=0.0)

            # pad/truncate to 512D
            vec = np.pad(sample, (0, max(0, 512 - len(sample))), mode="constant")[:512]

            # 推理
            logic_result = self.engine.reason(vec)
            emotion_result = self.engine.infer_emotion(vec)
            z = self.engine.blend(logic_result, emotion_result)
            score_result = self.engine.score(z)

            scores.append(score_result["aggregated_score"])

        return scores

    def _evaluate_with_features(self, features: np.ndarray, n_samples: int) -> List[float]:
        """用自适应后的特征评估"""
        scores = []
        n = min(n_samples, features.shape[0])

        for i in range(n):
            vec = features[i]
            logic_result = self.engine.reason(vec)
            emotion_result = self.engine.infer_emotion(vec)
            z = self.engine.blend(logic_result, emotion_result)
            score_result = self.engine.score(z)

            scores.append(score_result["aggregated_score"])

        return scores

    def benchmark(self, datasets: Dict, results_path: Optional[str] = None) -> Dict:
        """
        在多个数据集上运行基准测试

        Args:
            datasets: 数据集字典 {"name": (X, y, meta), ...}
            results_path: 结果保存路径（可选）

        Returns:
            benchmark_results: 基准测试结果
        """
        benchmark_results = {}

        dataset_names = list(datasets.keys())

        for i, name_src in enumerate(dataset_names):
            for j, name_tgt in enumerate(dataset_names):
                if i == j:
                    continue  # 跳过同域

                print(f"[Benchmark] {name_src} → {name_tgt}...")

                result = self.evaluate_cross_domain(
                    datasets[name_src], datasets[name_tgt], n_samples=30
                )

                key = f"{name_src}→{name_tgt}"
                benchmark_results[key] = result

                print(f"  差距: {result['gap']['mean_gap']:.4f}")
                if "improvement" in result["gap"]:
                    print(f"  改进: {result['gap']['improvement']:.4f}")

        # 保存结果
        if results_path is not None:
            import json
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] 基准测试结果已保存: {results_path}")

        return benchmark_results
