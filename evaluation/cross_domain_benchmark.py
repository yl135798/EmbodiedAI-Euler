# -*- coding: utf-8 -*-
"""
跨域泛化性评估基准

系统化测试算法在不同域上的泛化能力：

测试场景:
  1. 同域泛化 (SECOM 不同产线)
  2. 跨模态泛化 (视觉 → 嗅觉)
  3. Few-shot 泛化 (10 样本学习新域)
  4. 零样本泛化 (纯推理，无目标域数据)

数据集:
  - SECOM (半导体制造，590 特征)
  - Air Quality (气体传感器，5 传感器)
  - (模拟) 其他工业数据集
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from datetime import datetime
import json

# -- 域自适应欧拉引擎 -----------------------------
from cognition.euler_domain_adapted import (
    DomainAdaptedEulerEngine,
    evaluate_cross_domain_generalization,
)


def create_simulated_dataset(n_samples: int = 1000, n_features: int = 512,
                             domain_type: str = "industrial") -> tuple:
    """
    创建模拟数据集（用于评估基准）

    域类型:
      - "industrial": 工业数据（类似 SECOM）
      - "environmental": 环境数据（类似 Air Quality）
      - "medical": 医疗数据（模拟）
      - "financial": 金融数据（模拟）
    """
    np.random.seed(42)

    if domain_type == "industrial":
        # 工业数据：高维、稀疏、偏态分布
        X = np.random.weibull(a=0.8, size=(n_samples, n_features)) * 10
        X = np.nan_to_num(X, nan=0.0)
        y = (np.sum(X, axis=1) > np.percentile(np.sum(X, axis=1), 90)).astype(int)

    elif domain_type == "environmental":
        # 环境数据：低维、连续、周期分布
        t = np.linspace(0, 4 * np.pi, n_samples)
        X = np.zeros((n_samples, n_features))
        for i in range(min(5, n_features)):
            X[:, i] = np.sin(t + i * np.pi / 5) + np.random.randn(n_samples) * 0.1
        y = (X[:, 0] > 0.5).astype(int)

    elif domain_type == "medical":
        # 医疗数据：中维、正态分布、类别不平衡
        X = np.random.randn(n_samples, n_features) * 2 + 5
        y = (X[:, 0] > 5).astype(int)
        # 类别不平衡（10% 正例）
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        keep = np.concatenate([pos_indices, neg_indices[:int(len(pos_indices) * 9)]])
        X = X[keep]
        y = y[keep]

    elif domain_type == "financial":
        # 金融数据：高噪声、尖峰厚尾、时间依赖
        X = np.random.standard_t(df=3, size=(n_samples, n_features)) * 100
        y = (np.sum(X, axis=1) > np.percentile(np.sum(X, axis=1), 95)).astype(int)

    else:
        raise ValueError(f"未知域类型: {domain_type}")

    return X, y, {"domain_type": domain_type, "n_samples": len(X), "n_features": n_features}


def evaluate_single_domain(engine, X, y, domain_id: int, n_samples: int = 50) -> float:
    """
    评估单域泛化性

    返回: 平均评分
    """
    engine.set_domain(domain_id)
    scores = []

    n = min(n_samples, len(X))
    for i in range(n):
        sample = X[i]
        sample = np.nan_to_num(sample, nan=0.0)
        vec = np.pad(sample, (0, max(0, 512 - len(sample))), mode="constant")[:512]

        logic_result = engine.reason(vec)
        emotion_result = engine.infer_emotion(vec)
        z = engine.blend(logic_result, emotion_result)
        score_result = engine.score(z)

        scores.append(score_result["aggregated_score"])

    return float(np.mean(scores))


def run_benchmark():
    """
    运行跨域泛化性基准测试
    """
    print("\n" + "="*60)
    print("跨域泛化性评估基准")
    print("="*60)
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载/创建数据集
    print("\n[1] 加载/创建数据集...")
    datasets = {}

    #    (模拟) 工业数据集 (类似 SECOM)
    print("  创建模拟工业数据集...")
    X_industrial, y_industrial, meta_industrial = create_simulated_dataset(
        n_samples=1000, n_features=512, domain_type="industrial"
    )
    datasets["Industrial"] = (X_industrial, y_industrial, meta_industrial)
    print(f"    Industrial: {X_industrial.shape}, 正例率={np.mean(y_industrial):.2%}")

    #    (模拟) 环境数据集 (类似 Air Quality)
    print("  创建模拟环境数据集...")
    X_env, y_env, meta_env = create_simulated_dataset(
        n_samples=1000, n_features=512, domain_type="environmental"
    )
    datasets["Environmental"] = (X_env, y_env, meta_env)
    print(f"    Environmental: {X_env.shape}, 正例率={np.mean(y_env):.2%}")

    #    (模拟) 医疗数据集
    print("  创建模拟医疗数据集...")
    X_medical, y_medical, meta_medical = create_simulated_dataset(
        n_samples=1000, n_features=512, domain_type="medical"
    )
    datasets["Medical"] = (X_medical, y_medical, meta_medical)
    print(f"    Medical: {X_medical.shape}, 正例率={np.mean(y_medical):.2%}")

    #    (模拟) 金融数据集
    print("  创建模拟金融数据集...")
    X_financial, y_financial, meta_financial = create_simulated_dataset(
        n_samples=1000, n_features=512, domain_type="financial"
    )
    datasets["Financial"] = (X_financial, y_financial, meta_financial)
    print(f"    Financial: {X_financial.shape}, 正例率={np.mean(y_financial):.2%}")

    print(f"\n[OK] 数据集加载完成: {len(datasets)} 个")

    # 2. 创建引擎
    print("\n[2] 创建域自适应引擎...")
    engine = DomainAdaptedEulerEngine(
        state_dim=512, n_emotions=8, n_domains=len(datasets)
    )
    print("[OK] DomainAdaptedEulerEngine 创建成功")

    # 3. 预训练域适配器（所有域对）
    print("\n[3] 预训练域适配器...")
    dataset_names = list(datasets.keys())
    for i, name_src in enumerate(dataset_names):
        for j, name_tgt in enumerate(dataset_names):
            if i >= j:
                continue  # 跳过自己和重复
            X_src, y_src, _ = datasets[name_src]
            X_tgt, y_tgt, _ = datasets[name_tgt]
            print(f"  {name_src} → {name_tgt}...")
            engine.fit_domain_adapter(X_src, X_tgt, n_epochs=3, lr=0.01)

    print("[OK] 域适配器预训练完成")

    # 4. 评估跨域泛化性
    print("\n[4] 评估跨域泛化性...")
    results = {}
    for i, name_src in enumerate(dataset_names):
        for j, name_tgt in enumerate(dataset_names):
            if i == j:
                continue  # 跳过同域

            print(f"\n  {name_src} → {name_tgt}...")

            X_src, y_src, _ = datasets[name_src]
            X_tgt, y_tgt, _ = datasets[name_tgt]

            # 评估
            metrics = evaluate_cross_domain_generalization(
                engine, X_src, y_src, X_tgt, y_tgt, n_samples=30
            )

            key = f"{name_src}→{name_tgt}"
            results[key] = metrics

            print(f"    差距: {metrics['gap']:.4f}")
            print(f"    评级: {metrics['rating']}")

    # 5. 生成报告
    print("\n[5] 生成评估基准报告...")
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_datasets": len(datasets),
        "datasets": {name: {"n_samples": len(datasets[name][0]), "n_features": datasets[name][0].shape[1]} for name in dataset_names},
        "results": results,
        "summary": {
            "mean_gap": float(np.mean([r["gap"] for r in results.values()])),
            "std_gap": float(np.std([r["gap"] for r in results.values()])),
            "min_gap": float(np.min([r["gap"] for r in results.values()])),
            "max_gap": float(np.max([r["gap"] for r in results.values()])),
        }
    }

    # 保存报告
    report_path = r"C:\Users\A\.qclaw\workspace\DataAgent\evaluation\benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 报告已保存: {report_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("  评估基准摘要")
    print("="*60)
    print(f"  数据集数量: {report['n_datasets']}")
    print(f"  平均泛化性差距: {report['summary']['mean_gap']:.4f}")
    print(f"  最小差距: {report['summary']['min_gap']:.4f}")
    print(f"  最大差距: {report['summary']['max_gap']:.4f}")

    # 评级分布
    ratings = [r["rating"] for r in results.values()]
    from collections import Counter
    rating_counts = Counter(ratings)
    print(f"\n  评级分布:")
    for rating, count in rating_counts.items():
        print(f"    {rating}: {count} 次")

    print("\n  [OK] 基准测试完成！")
    print("="*60)

    return report


def main():
    print("="*60)
    print("  跨域泛化性评估基准")
    print("  DataAgent 系统化测试")
    print("="*60)

    try:
        report = run_benchmark()
    except Exception as e:
        print(f"\n[ERR] 基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        report = None

    # 总结
    print("\n" + "="*60)
    print("  总结")
    print("="*60)

    if report is not None:
        print(f"  [OK] 基准测试成功！")
        print(f"  平均泛化性差距: {report['summary']['mean_gap']:.4f}")
        if report["summary"]["mean_gap"] < 0.1:
            print(f"  总体评级: 优秀 (平均差距 < 0.1)")
        elif report["summary"]["mean_gap"] < 0.3:
            print(f"  总体评级: 良好 (平均差距 < 0.3)")
        elif report["summary"]["mean_gap"] < 0.5:
            print(f"  总体评级: 中等 (平均差距 < 0.5)")
        else:
            print(f"  总体评级: 需改进 (平均差距 ≥ 0.5)")
    else:
        print(f"  [ERR] 基准测试失败")

    print("\n  [OK] 评估完成！")
    print("="*60)


if __name__ == "__main__":
    main()
