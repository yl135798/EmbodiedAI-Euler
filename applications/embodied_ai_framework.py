# -*- coding: utf-8 -*-
"""
具身智能 - 统一认知框架
Embodied AI Unified Cognitive Framework

将欧拉同构认知模型部署为具身智能体的"大脑"，
实现 感知→认知→决策→行动 的完整闭环。
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class EmbodimentType(Enum):
    WHEELED_ROBOT = "wheeled"
    LEGGED_ROBOT = "legged"
    DRONE = "drone"
    MANIPULATOR = "manipulator"
    VIRTUAL_AGENT = "virtual"


class CognitiveMode(Enum):
    REACTIVE = "reactive"       # 反应式：快速响应，低思考
    DELIBERATIVE = "deliberative"  # 审慎式：深度推理，慢响应
    HYBRID = "hybrid"           # 混合式：根据情境自动切换


@dataclass
class EmbodiedState:
    """具身智能体的完整状态表示"""
    # 物理状态
    position: np.ndarray       # 3D 位置 [x, y, z]
    orientation: np.ndarray    # 四元数姿态 [qw, qx, qy, qz]
    velocity: np.ndarray       # 线速度 [vx, vy, vz]
    angular_vel: np.ndarray    # 角速度 [wx, wy, wz]
    energy: float              # 能量水平 [0, 1]
    
    # 认知状态 (欧拉复数表示)
    cognitive_z: complex = 0+0j  # 复数认知状态 z = r·e^(iθ)
    logic_state: float = 0.0     # cos(θ) 逻辑推理分量
    emotion_state: float = 0.0   # sin(θ) 情感感知分量
    confidence: float = 0.0      # |z| 置信度
    cognitive_phase: float = 0.0 # arg(z) 认知相位
    
    # 五感缓冲区
    sensory_buffer: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 行为状态
    current_action: Optional[str] = None
    action_queue: List[str] = field(default_factory=list)
    
    # 学习状态
    memory_trace: List[Dict] = field(default_factory=list)
    adaptation_rate: float = 0.01


class EmbodiedCognitiveBrain:
    """
    具身认知大脑 - 欧拉同构统一框架
    
    架构：
    五感输入 → 感知融合 → 欧拉认知引擎 → 行为决策 → 运动输出
                    ↑                                    ↓
                    └──── 学习与适应 (反馈闭环) ────────┘
    """
    
    def __init__(self, embodiment_type: EmbodimentType = EmbodimentType.WHEELED_ROBOT,
                 state_dim: int = 512, action_dim: int = 64):
        self.embodiment_type = embodiment_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 欧拉认知引擎核心参数
        self.W_logic = np.random.randn(state_dim, state_dim) * 0.01   # 逻辑权重
        self.W_emotion = np.random.randn(state_dim, state_dim) * 0.01 # 情感权重
        self.b_logic = np.zeros(state_dim)
        self.b_emotion = np.zeros(state_dim)
        
        # 认知模式参数
        self.cognitive_mode = CognitiveMode.HYBRID
        self.reactive_threshold = 0.3   # 威胁/紧急度阈值
        self.deliberative_threshold = 0.7  # 复杂度阈值
        
        # 具身状态
        self.state = EmbodiedState(
            position=np.zeros(3),
            orientation=np.array([1, 0, 0, 0]),
            velocity=np.zeros(3),
            angular_vel=np.zeros(3),
            energy=1.0
        )
        
        # 五感配置 (根据具身类型调整)
        self.sensor_configs = self._init_sensor_configs()
        
        # 行为库
        self.behavior_library = self._init_behavior_library()
        
        # 学习参数
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.experience_buffer: List[Dict] = []
        
        # 内在动机系统
        self.curiosity = 0.5       # 好奇心驱动
        self.safety_weight = 0.8   # 安全权重
        self.efficiency = 0.5      # 效率偏好
    
    def _init_sensor_configs(self) -> Dict:
        """根据具身类型配置传感器"""
        configs = {
            EmbodimentType.WHEELED_ROBOT: {
                "visual": {"dim": 512, "range": 10.0, "fov": 120},
                "auditory": {"dim": 256, "range": 5.0},
                "tactile": {"dim": 128, "zones": ["front", "rear", "left", "right"]},
                "olfactory": {"dim": 64, "range": 2.0},
                "proprioceptive": {"dim": 64, "joints": ["left_wheel", "right_wheel"]}
            },
            EmbodimentType.LEGGED_ROBOT: {
                "visual": {"dim": 512, "range": 8.0, "fov": 180},
                "auditory": {"dim": 256, "range": 5.0},
                "tactile": {"dim": 256, "zones": ["feet", "body", "head"]},
                "proprioceptive": {"dim": 128, "joints": ["hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"]}
            },
            EmbodimentType.DRONE: {
                "visual": {"dim": 512, "range": 50.0, "fov": 360},
                "auditory": {"dim": 128, "range": 10.0},
                "tactile": {"dim": 64, "zones": ["body"]},
                "proprioceptive": {"dim": 128, "joints": ["rotor_fl", "rotor_fr", "rotor_bl", "rotor_br"]}
            },
            EmbodimentType.MANIPULATOR: {
                "visual": {"dim": 512, "range": 2.0, "fov": 90},
                "tactile": {"dim": 256, "zones": ["fingertips", "palm", "wrist"]},
                "proprioceptive": {"dim": 128, "joints": ["shoulder", "elbow", "wrist", "gripper"]}
            },
            EmbodimentType.VIRTUAL_AGENT: {
                "visual": {"dim": 512, "range": float('inf'), "fov": 360},
                "auditory": {"dim": 256, "range": float('inf')},
                "tactile": {"dim": 128, "zones": ["virtual_body"]},
                "proprioceptive": {"dim": 64, "joints": ["virtual_skeleton"]}
            }
        }
        return configs.get(self.embodiment_type, configs[EmbodimentType.WHEELED_ROBOT])
    
    def _init_behavior_library(self) -> Dict:
        """初始化行为库"""
        base_behaviors = {
            "explore": {"energy_cost": 0.02, "safety": 0.8, "curiosity_gain": 0.1},
            "approach": {"energy_cost": 0.03, "safety": 0.6, "curiosity_gain": 0.05},
            "retreat": {"energy_cost": 0.02, "safety": 0.95, "curiosity_gain": 0.0},
            "observe": {"energy_cost": 0.005, "safety": 0.9, "curiosity_gain": 0.08},
            "manipulate": {"energy_cost": 0.05, "safety": 0.5, "curiosity_gain": 0.12},
            "communicate": {"energy_cost": 0.01, "safety": 0.9, "curiosity_gain": 0.03},
            "rest": {"energy_cost": -0.05, "safety": 1.0, "curiosity_gain": 0.0},
            "emergency_stop": {"energy_cost": 0.0, "safety": 1.0, "curiosity_gain": 0.0}
        }
        return base_behaviors
    
    # ==================== 感知层 ====================
    
    def perceive(self, sensory_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        五感融合：将多模态传感器输入统一编码
        
        Args:
            sensory_inputs: {"visual": array, "auditory": array, ...}
        
        Returns:
            fused: 融合后的感知向量
        """
        encoded = {}
        for modality, data in sensory_inputs.items():
            config = self.sensor_configs.get(modality, {"dim": 64})
            target_dim = config.get("dim", 64)
            
            if data.shape[0] > target_dim:
                encoded[modality] = data[:target_dim]
            elif data.shape[0] < target_dim:
                padded = np.zeros(target_dim)
                padded[:data.shape[0]] = data
                encoded[modality] = padded
            else:
                encoded[modality] = data
        
        # 加权融合 (视觉权重最高)
        modality_weights = {"visual": 0.35, "auditory": 0.20, "tactile": 0.20,
                           "olfactory": 0.10, "proprioceptive": 0.15}
        
        fused = np.zeros(self.state_dim)
        total_weight = 0.0
        for modality, data in encoded.items():
            w = modality_weights.get(modality, 0.1)
            if data.shape[0] <= self.state_dim:
                fused[:data.shape[0]] += w * data
            total_weight += w
        
        if total_weight > 0:
            fused /= total_weight
        
        self.state.sensory_buffer = encoded
        return {"fused": fused, "per_modality": encoded}
    
    # ==================== 认知层 ====================
    
    def think(self, perception: Dict[str, np.ndarray]) -> complex:
        """
        欧拉认知引擎：逻辑 + 情感 → 复数认知状态
        
        z = r · e^(iθ)
        其中 cos(θ) = 逻辑分量, sin(θ) = 情感分量
        """
        fused = perception["fused"]
        
        # 逻辑路径：符号推理 + 因果推断
        logic_raw = np.tanh(fused @ self.W_logic + self.b_logic)
        self.state.logic_state = float(np.mean(logic_raw))
        
        # 情感路径：Plutchik 八维情绪空间
        emotion_raw = np.tanh(fused @ self.W_emotion + self.b_emotion)
        self.state.emotion_state = float(np.mean(emotion_raw))
        
        # 欧拉同构融合
        r = np.sqrt(self.state.logic_state**2 + self.state.emotion_state**2 + 1e-8)
        theta = np.arctan2(self.state.emotion_state, self.state.logic_state)
        
        # 数值稳定的复数表示
        self.state.cognitive_z = r * cmath.exp(1j * theta)
        self.state.confidence = r
        self.state.cognitive_phase = theta
        
        return self.state.cognitive_z
    
    def select_cognitive_mode(self, situation_urgency: float, task_complexity: float) -> CognitiveMode:
        """
        根据情境选择认知模式
        
        - 紧急威胁 → 反应式 (快速但粗糙)
        - 高复杂度 → 审慎式 (慢但精确)
        - 其他 → 混合式
        """
        if situation_urgency > (1 - self.reactive_threshold):
            self.cognitive_mode = CognitiveMode.REACTIVE
        elif task_complexity > self.deliberative_threshold:
            self.cognitive_mode = CognitiveMode.DELIBERATIVE
        else:
            self.cognitive_mode = CognitiveMode.HYBRID
        
        return self.cognitive_mode
    
    def evaluate_situation(self, perception: Dict[str, np.ndarray]) -> Dict:
        """
        情境评估：威胁度 + 复杂度 + 机会度
        """
        fused = perception["fused"]
        
        urgency = float(1.0 / (1.0 + np.exp(-np.mean(np.abs(fused[:64])))))
        complexity = float(np.std(fused))
        opportunity = float(1.0 / (1.0 + np.exp(-np.mean(fused[64:128]))))
        
        self.select_cognitive_mode(urgency, complexity)
        
        return {
            "urgency": urgency,
            "complexity": complexity,
            "opportunity": opportunity,
            "cognitive_mode": self.cognitive_mode.value,
            "energy": self.state.energy,
            "curiosity": self.curiosity
        }
    
    # ==================== 决策层 ====================
    
    def decide(self, cognitive_z: complex, situation: Dict) -> Dict:
        """
        行为决策：基于认知状态和情境评估
        
        资源分配 = |z| (置信度 → 能量投入)
        权力分配 = arg(z) (认知相位 → 行为选择)
        """
        r = abs(cognitive_z)
        theta = cmath.phase(cognitive_z)
        
        # 内在动机驱动的行为评分
        behavior_scores = {}
        for behavior, meta in self.behavior_library.items():
            safety_score = meta["safety"] * self.safety_weight
            curiosity_score = meta["curiosity_gain"] * self.curiosity
            energy_factor = 1.0 if meta["energy_cost"] <= 0 else max(0, self.state.energy - meta["energy_cost"])
            efficiency_score = meta.get("efficiency", 0.5) * self.efficiency
            
            # 欧拉认知调制
            cognitive_modulation = r * np.cos(theta - self._behavior_phase(behavior))
            
            # 情境适配
            if situation["urgency"] > 0.7 and behavior == "emergency_stop":
                cognitive_modulation += 2.0
            
            total = safety_score + curiosity_score + efficiency_score + cognitive_modulation * energy_factor
            behavior_scores[behavior] = total
        
        # 选择最优行为
        best_behavior = max(behavior_scores, key=behavior_scores.get)
        
        # 资源分配
        resource_allocation = {
            "energy": min(r * 0.3, self.state.energy * 0.5),
            "attention": np.cos(theta) * r,
            "computation": np.sin(theta) * r,
        }
        
        # 权力分配 (决策权重)
        power_allocation = {
            "logic_weight": max(0, np.cos(theta)),
            "emotion_weight": max(0, np.sin(theta)),
            "safety_override": situation["urgency"] > 0.8
        }
        
        return {
            "selected_behavior": best_behavior,
            "behavior_scores": behavior_scores,
            "resource_allocation": resource_allocation,
            "power_allocation": power_allocation,
            "confidence": r,
            "phase": theta
        }
    
    def _behavior_phase(self, behavior: str) -> float:
        """将行为映射到认知相位空间"""
        phase_map = {
            "explore": np.pi / 4,      # 好奇+逻辑
            "approach": np.pi / 6,      # 偏逻辑
            "retreat": 3 * np.pi / 4,   # 偏情感(恐惧)
            "observe": np.pi / 3,       # 逻辑为主
            "manipulate": np.pi / 8,    # 高逻辑
            "communicate": np.pi / 2,   # 逻辑+情感平衡
            "rest": 5 * np.pi / 4,      # 情感主导(疲惫)
            "emergency_stop": np.pi,     # 纯情感(恐惧)
        }
        return phase_map.get(behavior, 0.0)
    
    # ==================== 执行层 ====================
    
    def act(self, decision: Dict) -> Dict:
        """
        执行行为并更新状态
        """
        behavior = decision["selected_behavior"]
        meta = self.behavior_library[behavior]
        
        # 消耗能量
        energy_cost = meta["energy_cost"] * decision["resource_allocation"]["energy"]
        self.state.energy = max(0, self.state.energy - abs(energy_cost))
        
        # 如果是休息，恢复能量
        if behavior == "rest":
            self.state.energy = min(1.0, self.state.energy + 0.1)
        
        # 更新好奇心
        self.curiosity = max(0, min(1, self.curiosity + meta["curiosity_gain"] - 0.02))
        
        # 生成运动指令
        motor_command = self._generate_motor_command(behavior, decision)
        
        # 记录经验
        experience = {
            "behavior": behavior,
            "confidence": decision["confidence"],
            "energy_before": self.state.energy + abs(energy_cost),
            "energy_after": self.state.energy,
            "cognitive_mode": self.cognitive_mode.value,
            "timestamp": time.time()
        }
        self.experience_buffer.append(experience)
        self.state.memory_trace.append(experience)
        
        self.state.current_action = behavior
        
        return {
            "motor_command": motor_command,
            "behavior": behavior,
            "energy_remaining": self.state.energy,
            "curiosity": self.curiosity
        }
    
    def _generate_motor_command(self, behavior: str, decision: Dict) -> np.ndarray:
        """根据行为类型生成运动指令"""
        base_command = np.zeros(self.action_dim)
        r = decision["confidence"]
        theta = decision["phase"]
        
        behavior_commands = {
            "explore": np.array([0.5*r, 0, 0, 0, 0.3*np.sin(theta)]),  # 前进+轻微转向
            "approach": np.array([0.8*r, 0, 0, 0, 0]),                   # 直线前进
            "retreat": np.array([-0.6*r, 0, 0, 0, 0.5*np.cos(theta)]), # 后退+转向
            "observe": np.array([0.1*r, 0, 0, 0, 0.2*np.sin(theta)]),   # 缓慢转动
            "manipulate": np.array([0, 0, 0.5*r, 0.3*r, 0]),           # 手臂动作
            "communicate": np.array([0]*5),                               # 无运动
            "rest": np.array([0]*5),                                      # 无运动
            "emergency_stop": np.array([0]*5),                            # 急停
        }
        
        cmd = behavior_commands.get(behavior, np.zeros(5))
        base_command[:len(cmd)] = cmd
        return base_command
    
    # ==================== 学习层 ====================
    
    def learn(self, reward: float, next_state_perception: Dict[str, np.ndarray]):
        """
        从经验中学习，更新认知权重
        
        使用欧拉梯度下降：
        ΔW = η · ∂R/∂z · ∂z/∂W
        """
        if len(self.experience_buffer) == 0:
            return
        
        last_exp = self.experience_buffer[-1]
        
        # TD 误差
        td_error = reward + self.discount_factor * abs(self.state.cognitive_z) - last_exp["confidence"]
        
        # 更新逻辑权重
        grad_logic = td_error * np.cos(self.state.cognitive_phase)
        self.W_logic += self.learning_rate * grad_logic * 0.01
        
        # 更新情感权重
        grad_emotion = td_error * np.sin(self.state.cognitive_phase)
        self.W_emotion += self.learning_rate * grad_emotion * 0.01
        
        # 自适应内在动机
        if reward > 0:
            self.curiosity *= 0.95  # 成功后降低好奇心
            self.safety_weight *= 0.98  # 逐渐敢于冒险
        else:
            self.curiosity = min(1.0, self.curiosity * 1.05)  # 失败增加好奇
            self.safety_weight = min(1.0, self.safety_weight * 1.02)  # 更趋安全
    
    # ==================== 完整闭环 ====================
    
    def step(self, sensory_inputs: Dict[str, np.ndarray], reward: float = 0.0) -> Dict:
        """
        完整的 感知→认知→决策→行动 闭环
        
        Args:
            sensory_inputs: 五感输入
            reward: 外部奖励信号
        
        Returns:
            完整的决策和执行结果
        """
        # 1. 感知
        perception = self.perceive(sensory_inputs)
        
        # 2. 情境评估
        situation = self.evaluate_situation(perception)
        
        # 3. 认知
        cognitive_z = self.think(perception)
        
        # 4. 决策
        decision = self.decide(cognitive_z, situation)
        
        # 5. 执行
        action_result = self.act(decision)
        
        # 6. 学习
        self.learn(reward, perception)
        
        return {
            "perception": perception,
            "situation": situation,
            "cognitive_state": {
                "z": cognitive_z,
                "logic": self.state.logic_state,
                "emotion": self.state.emotion_state,
                "confidence": self.state.confidence,
                "phase": self.state.cognitive_phase,
                "mode": self.cognitive_mode.value
            },
            "decision": decision,
            "action": action_result
        }
    
    def get_state_report(self) -> Dict:
        """获取完整的智能体状态报告"""
        return {
            "embodiment": self.embodiment_type.value,
            "energy": self.state.energy,
            "confidence": self.state.confidence,
            "cognitive_mode": self.cognitive_mode.value,
            "logic": self.state.logic_state,
            "emotion": self.state.emotion_state,
            "curiosity": self.curiosity,
            "safety_weight": self.safety_weight,
            "current_action": self.state.current_action,
            "experience_count": len(self.experience_buffer)
        }


# ==================== 演示 ====================

def demo_embodied_ai():
    """具身智能统一认知框架演示"""
    print("=" * 70)
    print("  具身智能 - 欧拉同构统一认知框架演示")
    print("  Embodied AI - Euler Isomorphism Unified Cognitive Framework")
    print("=" * 70)
    
    # 创建不同具身类型的智能体
    agents = {
        "轮式机器人": EmbodiedCognitiveBrain(EmbodimentType.WHEELED_ROBOT),
        "四足机器人": EmbodiedCognitiveBrain(EmbodimentType.LEGGED_ROBOT),
        "无人机": EmbodiedCognitiveBrain(EmbodimentType.DRONE),
        "机械臂": EmbodiedCognitiveBrain(EmbodimentType.MANIPULATOR),
    }
    
    for name, brain in agents.items():
        print(f"\n{'─' * 50}")
        print(f"  [{name}] 认知闭环测试")
        print(f"{'─' * 50}")
        
        # 模拟五感输入
        sensory = {
            "visual": np.random.randn(512) * 0.5,
            "auditory": np.random.randn(256) * 0.3,
            "tactile": np.random.randn(128) * 0.2,
            "proprioceptive": np.random.randn(64) * 0.1,
        }
        
        # 模拟动态环境交互
        for step in range(5):
            # 模拟环境变化
            if step == 2:
                # 突发威胁
                sensory["visual"] = np.random.randn(512) * 2.0
                sensory["auditory"] = np.random.randn(256) * 1.5
            
            result = brain.step(sensory, reward=np.random.uniform(-0.5, 1.0))
            
            cs = result["cognitive_state"]
            act = result["action"]
            
            print(f"  Step {step+1}: "
                  f"z={cs['z']:.4f} | "
                  f"logic={cs['logic']:.3f} emotion={cs['emotion']:.3f} | "
                  f"mode={cs['mode']:12s} | "
                  f"action={act['behavior']:15s} | "
                  f"energy={act['energy_remaining']:.2f} "
                  f"curiosity={act['curiosity']:.2f}")
    
    # 展示认知模式切换
    print(f"\n{'=' * 70}")
    print("  认知模式自适应切换测试")
    print(f"{'=' * 70}")
    
    brain = EmbodiedCognitiveBrain(EmbodimentType.WHEELED_ROBOT)
    
    scenarios = [
        ("正常巡逻", {"urgency": 0.2, "complexity": 0.3}),
        ("发现可疑物体", {"urgency": 0.5, "complexity": 0.6}),
        ("突发碰撞预警", {"urgency": 0.9, "complexity": 0.4}),
        ("复杂地形导航", {"urgency": 0.3, "complexity": 0.9}),
    ]
    
    for name, params in scenarios:
        mode = brain.select_cognitive_mode(params["urgency"], params["complexity"])
        print(f"  {name:12s} → 紧急度={params['urgency']:.1f} 复杂度={params['complexity']:.1f} → 模式: {mode.value}")


if __name__ == "__main__":
    demo_embodied_ai()
