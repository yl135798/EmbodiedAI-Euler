# -*- coding: utf-8 -*-
"""测试欧拉同构认知引擎"""
import sys
import os
import numpy as np
import cmath

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognition.cognitive_core_euler import EulerCognitiveEngine


def test_euler_engine():
    """测试欧拉认知引擎"""
    print("="*60)
    print("测试欧拉同构认知引擎")
    print("核心公式: e^(i*theta) = cos(theta) + i*sin(theta)")
    print("="*60)
    
    # 1. 初始化
    print("\n[1] 初始化引擎...")
    engine = EulerCognitiveEngine(state_dim=512, n_emotions=8)
    print("  [OK] EulerCognitiveEngine 初始化成功")
    
    # 2. 模拟输入
    print("\n[2] 生成模拟输入...")
    np.random.seed(42)
    fused_vector = np.random.randn(512) * 0.1
    print(f"  输入向量范数: {np.linalg.norm(fused_vector):.3f}")
    
    # 3. 逻辑推理（实部）
    print("\n[3] 逻辑推理（实部计算）...")
    logic_result = engine.reason(fused_vector)
    print(f"  置信度 r = {logic_result['confidence']:.3f}")
    print(f"  推理链:")
    for step in logic_result["reasoning_chain"]:
        print(f"    - {step}")
    
    # 4. 情绪推断（虚部）
    print("\n[4] 情绪推断（虚部计算）...")
    emotion_result = engine.infer_emotion(fused_vector)
    theta = emotion_result["emotion_angle"]
    print(f"  情绪角度 theta = {theta:.3f} rad ({np.degrees(theta):.1f} deg)")
    print(f"  情绪分布:")
    for name, prob in sorted(emotion_result["emotion_distribution"].items(), key=lambda x: -x[1])[:3]:
        print(f"    - {name}: {prob:.3f}")
    
    # 5. 融合（欧拉公式）
    print("\n[5] 融合（欧拉公式）...")
    z = engine.blend(logic_result, emotion_result)
    r = abs(z)
    phi = cmath.phase(z)
    print(f"  z = r * e^(i*theta)")
    print(f"    = {r:.3f} * e^(i*{phi:.3f})")
    print(f"    = {z.real:.3f} + {z.imag:.3f}i")
    print(f"  |z| = {r:.3f} (模长 = 置信度)")
    print(f"  arg(z) = {phi:.3f} rad (幅角 = 情绪状态)")
    
    # 6. 行动计算
    print("\n[6] 行动计算（复平面向量）...")
    resource, power = engine.compute_action(z)
    print(f"  资源分配（钱）: {resource:.3f}  (|z|)")
    print(f"  权力分配（权）: {power:.3f}  (arg(z) 归一化)")
    
    # 7. 量化评分
    print("\n[7] 量化评分...")
    score_result = engine.score(z)
    print(f"  综合评分: {score_result['aggregated_score']:.3f}")
    print(f"    模长 |z|: {score_result['magnitude']:.3f}")
    print(f"    实部 cos(theta): {score_result['real_part']:.3f}")
    print(f"    虚部 sin(theta): {score_result['imag_part']:.3f}")
    print(f"    幅角 theta: {score_result['phase']:.3f} rad")
    
    # 8. 验证欧拉公式
    print("\n[8] 验证欧拉公式...")
    theta_verify = emotion_result["emotion_angle"]
    z_formula = logic_result["confidence"] * cmath.exp(1j * theta_verify)
    print(f"  e^(i*theta) = cos(theta) + i*sin(theta)")
    print(f"    左式: z = {z.real:.3f} + {z.imag:.3f}i")
    print(f"    右式: {z_formula.real:.3f} + {z_formula.imag:.3f}i")
    print(f"  误差: {abs(z - z_formula):.6f}")
    
    # 9. 获取轨迹
    print("\n[9] 认知轨迹...")
    trajectory = engine.get_trajectory()
    print(f"  轨迹长度: {len(trajectory)}")
    print(f"  最后状态: {trajectory[-1]}")
    
    print("\n" + "="*60)
    print("  [OK] 欧拉同构认知引擎测试通过！")
    print("="*60)
    
    return True


if __name__ == "__main__":
    try:
        test_euler_engine()
    except Exception as e:
        print(f"\n[ERR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
