# EmbodiedAI-Euler

**基于欧拉恒等式同构的具身智能大脑框架**

> Five Senses → Knowledge (Logic + Emotion) → Action (Resource + Power) → Quantized Output

将欧拉恒等式 $e^{i\theta} = \cos(\theta) + i \cdot \sin(\theta)$ 同构映射为具身智能的认知架构，实现跨域零样本泛化。

---

## 🧠 核心理念

传统具身智能将感知、规划、控制分为独立模块，缺乏统一认知框架。本项目提出**欧拉同构认知模型**：

- **感知 (五感)** → 多模态传感器数据融合为复数向量
- **认知 (知)** → 复数 $z = r \cdot e^{i\theta}$ 同时编码逻辑与情感
- **决策 (行)** → 复平面上的向量 $(r, \theta)$ 映射为资源与权力分配
- **量化 (评分)** → $f(|z|, \arg(z))$ 输出可解释指标

### 欧拉恒等式 → 认知同构映射

| 欧拉分量 | 认知功能 | 物理含义 |
|---------|---------|---------|
| $\cos(\theta)$ | 逻辑推理 (Logic) | 符号推理、因果推断 |
| $\sin(\theta)$ | 情感感知 (Emotion) | Plutchik 八维情绪空间 |
| $r \cdot e^{i\theta}$ | 认知融合 (Fusion) | 复数乘法实现多源信息融合 |
| $(r, \theta)$ | 行为决策 (Action) | 资源分配 ($r$) + 权力分配 ($\theta$) |
| $\|z\| + \arg(z)$ | 量化评分 (Score) | 置信度 + 认知状态向量 |

---

## 📁 项目结构

```
EmbodiedAI-Euler/
├── perception/                  # 🔵 感知层 (五感输入)
│   └── sensor_fusion.py         # 多模态传感器融合 (视觉/听觉/触觉/嗅觉/味觉)
│
├── cognition/                   # 🟢 认知层 (知 - 逻辑+情感)
│   ├── cognitive_core.py        # 基础认知核心
│   ├── cognitive_core_euler.py  # ⭐ 欧拉认知引擎 (数值稳定 Softmax)
│   ├── domain_adapter.py        # MMD 域适应对齐
│   └── euler_domain_adapted.py  # 域适应包装器 (泛化差距 99.9%↓)
│
├── decision/                    # 🟡 决策层 (行 - 资源+权力)
│   └── executor.py              # 资源分配器 + 元权力建模
│
├── quantization/                # 🔴 量化层 (评分输出)
│   └── scorer.py                # 可解释量化评分系统
│
├── evaluation/                  # 📊 评估基准
│   ├── sota_comparison_v2.py    # SOTA 对比 (5种方法)
│   ├── cross_domain_benchmark.py
│   └── benchmark_report.json
│
├── theory/                      # 📐 数学理论
│   └── euler_convergence_and_mmd_bound.md  # 收敛性+MMD上界证明
│
└── demo*.py                     # 演示脚本
```

---

## 🏆 性能表现

### SOTA 跨域泛化对比 (10 数据集)

| 排名 | 方法 | 平均泛化差距 |
|:---:|------|:----------:|
| 🥇 | **Ours (Euler + MMD)** | **0.0086** |
| 🥈 | MAML | 0.0369 |
| 🥉 | DANN | 0.0902 |
| 4 | Reptile | 0.0925 |
| 5 | CDAN | 0.1005 |

> 泛化差距从 3.6568 降至 0.0042 (**99.9% 改进**)

---

## 🤖 扩展应用

### 1. 具身智能 (Embodied AI)

将 DataAgent 部署为机器人/虚拟体的"大脑"，通过五感融合实现：
- 环境感知与理解
- 认知推理与情感建模
- 自主行为决策
- 持续学习与适应

欧拉同构框架为具身智能提供了**统一的逻辑-情感融合范式**，区别于传统纯理性规划方法。

### 2. 机器人导航 (Robot Navigation)

```
激光雷达 + 视觉 + IMU → 五感融合 → 欧拉认知引擎 → 导航决策
```

- 多传感器融合：LiDAR、摄像头、IMU、超声波、触觉
- 跨环境零样本迁移：室内→室外、工厂→仓库、地面→楼梯
- 实时路径规划与障碍物规避
- 动态风险评估（$|z|$ 置信度 + $\arg(z)$ 方向）

### 3. 自动驾驶 (Autonomous Driving)

```
摄像头 + 毫米波雷达 + 激光雷达 + GPS → 欧拉引擎 → 驾驶决策
```

- 多模态感知融合（摄像头 + Radar + LiDAR + GPS）
- 跨条件适应：晴天→雨天→夜间→雾天
- 跨地域迁移：城市→高速→乡村→施工路段
- 实时量化风险评估与决策置信度输出

### 4. 工业质检 (Industrial Quality Inspection)

- 跨产线泛化：半导体→汽车→电子→食品
- SECOM 数据集验证：1567 样本 × 590 特征，6.6% 不合格率
- 多传感器联合检测（视觉+振动+温度+声学）

### 5. 环境监测 (Environmental Monitoring)

- Air Quality UCI 数据集验证：9357 样本，5 种气体传感器
- 跨站点迁移，无需重新标定

---

## 🚀 快速开始

```bash
pip install numpy scipy ucimlrepo

# 运行欧拉认知引擎演示
python demo_euler_domain_adapted.py

# 运行 SOTA 对比基准
python evaluation/sota_comparison_v2.py
```

---

## 📐 理论基础

详见 [`theory/euler_convergence_and_mmd_bound.md`](theory/euler_convergence_and_mmd_bound.md)

1. **欧拉认知引擎收敛性证明** — 复数权重梯度下降收敛
2. **MMD 域适应上界分析** — 最大均值差异的理论保证
3. **跨域泛化性理论** — 源域-目标域差距的量化上界

---

## 📊 数据集

| 数据集 | 样本数 | 特征数 | 领域 |
|--------|--------|--------|------|
| SECOM | 1,567 | 590 | 半导体制造 |
| Air Quality UCI | 9,357 | 17 | 环境监测 |
| Iris | 150 | 4 | 植物分类 |
| Digits | 1,797 | 64 | 手写识别 |
| Breast Cancer | 569 | 30 | 医学诊断 |

---

## License

MIT
