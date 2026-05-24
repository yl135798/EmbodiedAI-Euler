# -*- coding: utf-8 -*-
"""测试 DomainAdaptedEulerEngine (修复版)"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# 测试 DomainAdaptedEulerEngine
from cognition.euler_domain_adapted import DomainAdaptedEulerEngine, evaluate_cross_domain_generalization

print("="*60)
print("测试 DomainAdaptedEulerEngine (修复版)")
print("="*60)

# 1. 创建引擎 (不传 input_dim，默认使用 state_dim)
print("\n[1] 创建引擎...")
engine = DomainAdaptedEulerEngine(
    state_dim=512, n_emotions=8, n_domains=2
)  # ✅ 修复：不传 input_dim
print("[OK] DomainAdaptedEulerEngine 创建成功")

# 2. 生成模拟数据 (输入维度 = 512，与 state_dim 一致)
print("\n[2] 生成模拟数据...")
np.random.seed(42)
X_src = np.random.randn(100, 512)  # ✅ 修复：输入维度 = 512
X_tgt = np.random.randn(80, 512)
y_src = np.random.randint(0, 2, 100)
y_tgt = np.random.randint(0, 2, 80)
print(f"  [OK] 源域: {X_src.shape}, 目标域: {X_tgt.shape}")

# 3. 预训练域适配器
print("\n[3] 预训练域适配器...")
engine.fit_domain_adapter(X_src, X_tgt, n_epochs=3)
print("[OK] 域适配器预训练完成")

# 4. 评估跨域泛化性
print("\n[4] 评估跨域泛化性...")
metrics = evaluate_cross_domain_generalization(
    engine, X_src, y_src, X_tgt, y_tgt, n_samples=20
)

print(f"\n[结果] 跨域泛化性差距: {metrics['gap']:.4f}")
print(f"  评级: {metrics['rating']}")

# 5. 测试保存/加载
print("\n[5] 测试保存/加载...")
engine.save(r"C:\Users\A\.qclaw\workspace\DataAgent\test_model.npz")
engine.load(r"C:\Users\A\.qclaw\workspace\DataAgent\test_model.npz")
print("[OK] 保存/加载测试通过")

print("\n" + "="*60)
print("  [OK] DomainAdaptedEulerEngine 测试通过！")
print("="*60)
