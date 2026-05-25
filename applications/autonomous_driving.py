# -*- coding: utf-8 -*-
"""
自动驾驶 - 多模态感知+跨条件适应
Autonomous Driving - Multi-Modal Perception + Cross-Condition Adaptation

基于欧拉同构认知引擎，实现自动驾驶的多模态感知融合
和跨天气/光照/地域条件的零样本适应。
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class DrivingCondition(Enum):
    """驾驶条件"""
    DAY_SUNNY = "day_sunny"
    DAY_CLOUDY = "day_cloudy"
    DAY_RAINY = "day_rainy"
    NIGHT_DRY = "night_dry"
    NIGHT_WET = "night_wet"
    FOGGY = "foggy"
    SNOWY = "snowy"
    DUSK = "dusk"


class RoadType(Enum):
    """道路类型"""
    HIGHWAY = "highway"
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    CONSTRUCTION = "construction"
    PARKING_LOT = "parking_lot"


class DrivingAction(Enum):
    """驾驶动作"""
    MAINTAIN = "maintain"         # 保持
    ACCELERATE = "accelerate"     # 加速
    BRAKE = "brake"               # 刹车
    TURN_LEFT = "turn_left"       # 左转
    TURN_RIGHT = "turn_right"     # 右转
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    EMERGENCY_STOP = "emergency_stop"
    YIELD = "yield"               # 让行
    OVERTAKE = "overtake"         # 超车


@dataclass
class VehicleState:
    """车辆状态"""
    position: np.ndarray       # [x, y] (UTM or local)
    heading: float             # 航向角 (弧度)
    speed: float               # 速度 (m/s)
    acceleration: float        # 加速度 (m/s²)
    yaw_rate: float            # 横摆角速度 (rad/s)
    gear: int                  # 档位
    steering_angle: float      # 方向盘转角 (弧度)


@dataclass
class TrafficParticipant:
    """交通参与者"""
    track_id: int
    position: np.ndarray       # [x, y]
    velocity: np.ndarray       # [vx, vy]
    heading: float
    vehicle_type: str          # car, truck, pedestrian, cyclist, motorcycle
    distance: float            # 到自车的距离
    threat_level: float        # 威胁等级 [0, 1]


@dataclass
class DrivingCommand:
    """驾驶指令"""
    action: DrivingAction
    target_speed: float
    steering: float            # [-1, 1]
    confidence: float
    risk_level: float
    reasoning: str


class AutonomousDrivingBrain:
    """
    自动驾驶大脑 - 欧拉同构多模态认知
    
    核心能力：
    1. 摄像头 + 毫米波雷达 + 激光雷达 + GPS 多模态融合
    2. 跨天气/光照条件零样本适应
    3. 欧拉认知驱动的安全决策
    4. 实时量化风险评估
    """
    
    def __init__(self, state_dim: int = 512):
        self.state_dim = state_dim
        
        # 多模态感知编码器
        self.W_camera = np.random.randn(state_dim, 256) * 0.01
        self.W_lidar = np.random.randn(state_dim, 256) * 0.01
        self.W_radar = np.random.randn(state_dim, 128) * 0.01
        self.W_gps = np.random.randn(state_dim, 64) * 0.01
        
        # 欧拉认知引擎
        self.W_logic = np.random.randn(256, 128) * 0.01
        self.W_emotion = np.random.randn(256, 128) * 0.01
        
        # 驾驶条件原型（跨条件迁移关键）
        self.condition_prototypes: Dict[DrivingCondition, np.ndarray] = {}
        self._init_condition_prototypes()
        
        # 道路类型原型
        self.road_prototypes: Dict[RoadType, np.ndarray] = {}
        self._init_road_prototypes()
        
        # 认知状态
        self.cognitive_z = 0 + 0j
        self.driving_confidence = 0.5
        self.current_condition = DrivingCondition.DAY_SUNNY
        self.current_road = RoadType.URBAN
        
        # 安全参数
        self.safe_following_distance = 2.0   # 安全跟车距离 (秒)
        self.min_ttc = 3.0                    # 最小碰撞时间 (秒)
        self.max_deceleration = 8.0           # 最大减速度 (m/s²)
        self.comfort_deceleration = 3.0       # 舒适减速度 (m/s²)
        
        # 适应参数
        self.condition_adaptation_rate = 0.1
        self.perception_fusion_weights = {
            "camera": 0.35, "lidar": 0.30, "radar": 0.25, "gps": 0.10
        }
        
        # 经验缓冲
        self.driving_history: List[Dict] = []
        self.condition_performance: Dict[str, List[float]] = {}
    
    def _init_condition_prototypes(self):
        """初始化驾驶条件特征原型"""
        rng = np.random.RandomState(42)
        self.condition_prototypes = {
            DrivingCondition.DAY_SUNNY: rng.randn(self.state_dim) * 0.2 + 1.0,
            DrivingCondition.DAY_CLOUDY: rng.randn(self.state_dim) * 0.3 + 0.7,
            DrivingCondition.DAY_RAINY: rng.randn(self.state_dim) * 0.4 + 0.4,
            DrivingCondition.NIGHT_DRY: rng.randn(self.state_dim) * 0.3 - 0.3,
            DrivingCondition.NIGHT_WET: rng.randn(self.state_dim) * 0.5 - 0.5,
            DrivingCondition.FOGGY: rng.randn(self.state_dim) * 0.6 + 0.1,
            DrivingCondition.SNOWY: rng.randn(self.state_dim) * 0.5 + 0.2,
            DrivingCondition.DUSK: rng.randn(self.state_dim) * 0.35 + 0.3,
        }
    
    def _init_road_prototypes(self):
        """初始化道路类型特征原型"""
        rng = np.random.RandomState(43)
        self.road_prototypes = {
            RoadType.HIGHWAY: rng.randn(self.state_dim) * 0.3 + 0.8,
            RoadType.URBAN: rng.randn(self.state_dim) * 0.4 + 0.5,
            RoadType.SUBURBAN: rng.randn(self.state_dim) * 0.35 + 0.4,
            RoadType.RURAL: rng.randn(self.state_dim) * 0.5 + 0.2,
            RoadType.CONSTRUCTION: rng.randn(self.state_dim) * 0.6 - 0.1,
            RoadType.PARKING_LOT: rng.randn(self.state_dim) * 0.3 + 0.1,
        }
    
    # ==================== 多模态感知融合 ====================
    
    def fuse_perception(self, camera: np.ndarray, lidar: np.ndarray,
                        radar: np.ndarray, gps: np.ndarray,
                        condition: Optional[DrivingCondition] = None) -> Dict:
        """
        多模态传感器融合
        
        摄像头: 语义信息（车道线、交通标志、行人）
        LiDAR: 精确3D距离（障碍物、车辆位置）
        Radar: 远距离速度（移动车辆、天气穿透）
        GPS: 全局定位
        
        融合策略：根据当前条件动态调整各传感器权重
        """
        # 识别当前驾驶条件（如未提供）
        if condition is None:
            raw_percept = np.zeros(self.state_dim)
            raw_percept[:len(camera)] = camera[:min(len(camera), self.state_dim)]
            condition, _ = self.identify_condition(raw_percept)
        
        # 根据条件调整融合权重
        weights = self._adaptive_fusion_weights(condition)
        
        # 各传感器独立编码
        camera_feat = np.zeros(self.state_dim)
        cam_data = np.tanh(camera[:min(len(camera), 256)])
        camera_feat[:len(cam_data)] = cam_data
        
        lidar_feat = np.zeros(self.state_dim)
        lid_data = np.tanh(lidar[:min(len(lidar), 256)])
        lidar_feat[:len(lid_data)] = lid_data
        
        radar_feat = np.zeros(self.state_dim)
        rad_data = np.tanh(radar[:min(len(radar), 128)])
        radar_feat[:len(rad_data)] = rad_data
        
        gps_feat = np.zeros(self.state_dim)
        gps_data = np.tanh(gps[:min(len(gps), 64)])
        gps_feat[:len(gps_data)] = gps_data
        
        # 加权融合
        fused = (weights["camera"] * camera_feat + 
                weights["lidar"] * lidar_feat + 
                weights["radar"] * radar_feat + 
                weights["gps"] * gps_feat)
        
        return {
            "fused_perception": fused,
            "camera_features": camera_feat,
            "lidar_features": lidar_feat,
            "radar_features": radar_feat,
            "gps_features": gps_feat,
            "fusion_weights": weights,
            "detected_condition": condition
        }
    
    def _adaptive_fusion_weights(self, condition: DrivingCondition) -> Dict:
        """
        根据驾驶条件自适应调整传感器融合权重
        
        关键洞察：
        - 雨天/雾天 → 摄像头降权，雷达升权（穿透性好）
        - 夜间 → 摄像头降权，LiDAR升权（主动光源）
        - 晴天 → 摄像头最高权重（语义信息最丰富）
        """
        base = self.perception_fusion_weights.copy()
        
        adjustments = {
            DrivingCondition.DAY_SUNNY: {"camera": 1.0, "lidar": 1.0, "radar": 1.0, "gps": 1.0},
            DrivingCondition.DAY_CLOUDY: {"camera": 0.9, "lidar": 1.0, "radar": 1.1, "gps": 1.0},
            DrivingCondition.DAY_RAINY: {"camera": 0.5, "lidar": 0.8, "radar": 1.5, "gps": 1.0},
            DrivingCondition.NIGHT_DRY: {"camera": 0.4, "lidar": 1.4, "radar": 1.2, "gps": 1.1},
            DrivingCondition.NIGHT_WET: {"camera": 0.3, "lidar": 1.3, "radar": 1.6, "gps": 1.1},
            DrivingCondition.FOGGY: {"camera": 0.2, "lidar": 0.9, "radar": 1.8, "gps": 1.2},
            DrivingCondition.SNOWY: {"camera": 0.4, "lidar": 0.7, "radar": 1.4, "gps": 1.1},
            DrivingCondition.DUSK: {"camera": 0.6, "lidar": 1.2, "radar": 1.2, "gps": 1.0},
        }
        
        adj = adjustments.get(condition, {"camera": 1.0, "lidar": 1.0, "radar": 1.0, "gps": 1.0})
        
        adjusted = {k: base[k] * adj[k] for k in base}
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    # ==================== 条件识别与适应 ====================
    
    def identify_condition(self, perception: np.ndarray) -> Tuple[DrivingCondition, float]:
        """识别当前驾驶条件"""
        best_cond = DrivingCondition.DAY_SUNNY
        best_score = -float('inf')
        
        for cond, prototype in self.condition_prototypes.items():
            min_len = min(len(perception), len(prototype))
            sim = np.corrcoef(perception[:min_len], prototype[:min_len])[0, 1]
            if np.isnan(sim):
                sim = 0.0
            if sim > best_score:
                best_score = sim
                best_cond = cond
        
        self.current_condition = best_cond
        return best_cond, best_score
    
    def cross_condition_adapt(self, perception: np.ndarray,
                              source_condition: DrivingCondition,
                              target_condition: DrivingCondition) -> np.ndarray:
        """
        跨条件域适应
        
        将源条件下的感知映射到目标条件空间。
        核心思想：不同条件下同一场景的"不变特征"应该对齐。
        
        方法：欧拉旋转对齐 + MMD 域偏移补偿
        """
        src_proto = self.condition_prototypes[source_condition]
        tgt_proto = self.condition_prototypes[target_condition]
        
        min_len = min(len(perception), len(src_proto), len(tgt_proto))
        
        # 计算条件间的欧拉旋转角
        z_src = src_proto[:min_len] + 1j * np.zeros(min_len)
        z_tgt = tgt_proto[:min_len] + 1j * np.zeros(min_len)
        
        # 域偏移向量
        shift = z_tgt - z_src
        
        # 计算旋转角（取平均相位偏移）
        rotation_angle = np.angle(np.mean(shift)) * self.condition_adaptation_rate
        
        # 欧拉旋转
        rotation = cmath.exp(1j * rotation_angle)
        z_percept = perception[:min_len].astype(complex)
        z_adapted = z_percept * rotation
        
        # MMD 补偿（平移对齐均值）
        src_mean = np.mean(src_proto[:min_len])
        tgt_mean = np.mean(tgt_proto[:min_len])
        mean_shift = (tgt_mean - src_mean) * self.condition_adaptation_rate
        
        adapted = z_adapted.real + mean_shift
        
        # 补齐长度
        full_adapted = perception.copy()
        full_adapted[:min_len] = adapted
        
        return full_adapted
    
    # ==================== 认知与决策 ====================
    
    def driving_cognition(self, fused_perception: np.ndarray,
                          vehicle_state: VehicleState,
                          traffic: List[TrafficParticipant]) -> Dict:
        """
        欧拉认知引擎：驾驶场景理解与风险评估
        """
        # 逻辑路径：规则推理
        logic_input = fused_perception[:min(len(fused_perception), 128)]
        logic_raw = np.tanh(logic_input)
        logic_score = float(np.mean(logic_raw))
        
        # 情感路径：风险评估
        threat_scores = [p.threat_level for p in traffic] if traffic else [0.0]
        emotion_score = min(1.0, np.mean(threat_scores) * 2)
        
        # 欧拉融合
        r = np.sqrt(logic_score**2 + emotion_score**2 + 1e-8)
        theta = np.arctan2(emotion_score, logic_score)
        
        self.cognitive_z = r * cmath.exp(1j * theta)
        self.driving_confidence = abs(self.cognitive_z)
        
        # 安全评估
        ttc = self._compute_min_ttc(vehicle_state, traffic)
        safety_margin = self._compute_safety_margin(vehicle_state, traffic)
        
        return {
            "cognitive_z": self.cognitive_z,
            "logic_score": logic_score,
            "emotion_score": emotion_score,
            "confidence": r,
            "risk_phase": theta,
            "min_ttc": ttc,
            "safety_margin": safety_margin,
            "threat_count": sum(1 for p in traffic if p.threat_level > 0.5),
            "condition": self.current_condition.value
        }
    
    def _compute_min_ttc(self, ego: VehicleState, 
                         traffic: List[TrafficParticipant]) -> float:
        """计算最小碰撞时间 (TTC)"""
        if not traffic:
            return float('inf')
        
        min_ttc = float('inf')
        for p in traffic:
            if p.distance < 0.1:
                continue
            # 简化 TTC
            relative_speed = ego.speed - np.linalg.norm(p.velocity)
            if relative_speed > 0:
                ttc = p.distance / relative_speed
                min_ttc = min(min_ttc, ttc)
        
        return min_ttc
    
    def _compute_safety_margin(self, ego: VehicleState,
                               traffic: List[TrafficParticipant]) -> float:
        """计算安全裕度"""
        if not traffic:
            return 1.0
        
        min_dist = min(p.distance for p in traffic)
        safe_dist = ego.speed * self.safe_following_distance
        
        return max(0, min(1, min_dist / safe_dist))
    
    def make_driving_decision(self, cognition: Dict,
                               vehicle_state: VehicleState,
                               traffic: List[TrafficParticipant],
                               route_info: Dict = None) -> DrivingCommand:
        """
        基于欧拉认知状态的驾驶决策
        
        逻辑分量 → 规则遵从（交规、车道保持）
        情感分量 → 风险规避（安全距离、紧急制动）
        """
        z = cognition["cognitive_z"]
        r = abs(z)
        theta = cmath.phase(z)
        
        # 基于认知相位选择动作
        ttc = cognition["min_ttc"]
        safety = cognition["safety_margin"]
        
        # 决策逻辑
        if ttc < 1.0:
            action = DrivingAction.EMERGENCY_STOP
            target_speed = 0.0
            steering = 0.0
            confidence = r * 0.9
            risk = 1.0
            reasoning = f"紧急制动: TTC={ttc:.1f}s"
        elif ttc < self.min_ttc:
            action = DrivingAction.BRAKE
            target_speed = max(0, vehicle_state.speed * safety)
            steering = 0.0
            confidence = r * 0.7
            risk = 1 - ttc / self.min_ttc
            reasoning = f"减速避让: TTC={ttc:.1f}s, safety={safety:.2f}"
        elif safety < 0.3:
            action = DrivingAction.YIELD
            target_speed = vehicle_state.speed * 0.5
            steering = 0.0
            confidence = r * 0.6
            risk = 0.7
            reasoning = f"让行: safety_margin={safety:.2f}"
        elif route_info and route_info.get("turn_direction") == "left":
            action = DrivingAction.TURN_LEFT
            target_speed = vehicle_state.speed * 0.7
            steering = -0.5
            confidence = r * 0.8
            risk = 0.2
            reasoning = "按路线左转"
        elif route_info and route_info.get("turn_direction") == "right":
            action = DrivingAction.TURN_RIGHT
            target_speed = vehicle_state.speed * 0.7
            steering = 0.5
            confidence = r * 0.8
            risk = 0.2
            reasoning = "按路线右转"
        elif cognition["threat_count"] > 2:
            action = DrivingAction.BRAKE
            target_speed = vehicle_state.speed * 0.6
            steering = 0.0
            confidence = r * 0.5
            risk = 0.5
            reasoning = f"多威胁减速: {cognition['threat_count']}个威胁"
        else:
            action = DrivingAction.MAINTAIN
            target_speed = min(vehicle_state.speed + 0.5, 30.0)  # 限速30m/s
            steering = 0.0
            confidence = r
            risk = 0.1
            reasoning = "正常行驶"
        
        # 条件修正
        condition_speed_limits = {
            DrivingCondition.DAY_RAINY: 0.7,
            DrivingCondition.NIGHT_WET: 0.6,
            DrivingCondition.FOGGY: 0.5,
            DrivingCondition.SNOWY: 0.4,
        }
        limit = condition_speed_limits.get(self.current_condition, 1.0)
        target_speed *= limit
        
        return DrivingCommand(
            action=action,
            target_speed=target_speed,
            steering=steering,
            confidence=confidence,
            risk_level=risk,
            reasoning=reasoning
        )
    
    # ==================== 完整驾驶闭环 ====================
    
    def driving_step(self, sensor_data: Dict, vehicle_state: VehicleState,
                     traffic: List[TrafficParticipant],
                     route_info: Dict = None) -> Dict:
        """
        完整的自动驾驶决策闭环
        
        感知融合 → 条件识别 → 域适应 → 认知评估 → 驾驶决策
        """
        # 1. 多模态融合
        perception = self.fuse_perception(
            camera=sensor_data.get("camera", np.random.randn(512) * 0.3),
            lidar=sensor_data.get("lidar", np.random.randn(256) * 0.3),
            radar=sensor_data.get("radar", np.random.randn(128) * 0.2),
            gps=sensor_data.get("gps", np.random.randn(64) * 0.1)
        )
        
        # 2. 跨条件适应（如果检测到条件变化）
        adapted_perception = perception["fused_perception"]
        if perception["detected_condition"] != self.current_condition:
            adapted_perception = self.cross_condition_adapt(
                perception["fused_perception"],
                self.current_condition,
                perception["detected_condition"]
            )
        
        # 3. 认知评估
        cognition = self.driving_cognition(adapted_perception, vehicle_state, traffic)
        
        # 4. 驾驶决策
        command = self.make_driving_decision(cognition, vehicle_state, traffic, route_info)
        
        # 5. 记录经验
        self.driving_history.append({
            "condition": self.current_condition.value,
            "speed": vehicle_state.speed,
            "action": command.action.value,
            "confidence": command.confidence,
            "risk": command.risk_level,
            "ttc": cognition["min_ttc"],
            "timestamp": time.time()
        })
        
        return {
            "command": command,
            "cognition": cognition,
            "perception": perception,
            "fusion_weights": perception["fusion_weights"],
            "condition": self.current_condition.value,
            "condition_confidence": perception.get("detected_condition", self.current_condition)
        }


# ==================== 跨条件适应测试 ====================

def demo_autonomous_driving():
    """自动驾驶多模态感知+跨条件适应演示"""
    print("=" * 70)
    print("  自动驾驶 - 多模态感知+跨条件适应演示")
    print("  Autonomous Driving - Multi-Modal Perception + Cross-Condition")
    print("=" * 70)
    
    brain = AutonomousDrivingBrain()
    
    # 不同驾驶条件的场景
    conditions_scenarios = [
        {
            "condition": DrivingCondition.DAY_SUNNY,
            "name": "晴天白天",
            "vehicle": VehicleState(
                position=np.array([100.0, 50.0]), heading=0.0,
                speed=15.0, acceleration=0.0, yaw_rate=0.0,
                gear=3, steering_angle=0.0
            ),
            "traffic": [
                TrafficParticipant(1, np.array([105.0, 50.0]), np.array([14.0, 0.0]),
                                  0.0, "car", 5.0, 0.1),
                TrafficParticipant(2, np.array([95.0, 48.0]), np.array([0.0, 0.5]),
                                  np.pi/2, "pedestrian", 8.0, 0.3),
            ]
        },
        {
            "condition": DrivingCondition.DAY_RAINY,
            "name": "雨天白天",
            "vehicle": VehicleState(
                position=np.array([100.0, 50.0]), heading=0.0,
                speed=12.0, acceleration=0.0, yaw_rate=0.0,
                gear=3, steering_angle=0.0
            ),
            "traffic": [
                TrafficParticipant(3, np.array([103.0, 50.0]), np.array([10.0, 0.0]),
                                  0.0, "car", 3.0, 0.4),  # 更近，更危险
            ]
        },
        {
            "condition": DrivingCondition.NIGHT_WET,
            "name": "夜间湿滑",
            "vehicle": VehicleState(
                position=np.array([100.0, 50.0]), heading=0.0,
                speed=10.0, acceleration=0.0, yaw_rate=0.0,
                gear=2, steering_angle=0.0
            ),
            "traffic": [
                TrafficParticipant(4, np.array([102.0, 49.0]), np.array([8.0, 0.0]),
                                  0.0, "car", 2.0, 0.6),  # 夜间+湿滑，高风险
            ]
        },
        {
            "condition": DrivingCondition.FOGGY,
            "name": "大雾天气",
            "vehicle": VehicleState(
                position=np.array([100.0, 50.0]), heading=0.0,
                speed=8.0, acceleration=0.0, yaw_rate=0.0,
                gear=2, steering_angle=0.0
            ),
            "traffic": [
                TrafficParticipant(5, np.array([104.0, 50.0]), np.array([6.0, 0.0]),
                                  0.0, "truck", 4.0, 0.5),
            ]
        },
        {
            "condition": DrivingCondition.SNOWY,
            "name": "雪天行驶",
            "vehicle": VehicleState(
                position=np.array([100.0, 50.0]), heading=0.0,
                speed=6.0, acceleration=0.0, yaw_rate=0.0,
                gear=1, steering_angle=0.0
            ),
            "traffic": []  # 雪天路宽车少
        },
    ]
    
    for scenario in conditions_scenarios:
        print(f"\n{'─' * 60}")
        print(f"  条件: {scenario['name']} ({scenario['condition'].value})")
        print(f"  车速: {scenario['vehicle'].speed:.1f} m/s")
        print(f"  交通参与者: {len(scenario['traffic'])}")
        print(f"{'─' * 60}")
        
        # 生成条件相关的传感器数据
        cond = scenario["condition"]
        sensor_noise = {
            DrivingCondition.DAY_SUNNY: 0.1,
            DrivingCondition.DAY_RAINY: 0.3,
            DrivingCondition.NIGHT_WET: 0.4,
            DrivingCondition.FOGGY: 0.6,
            DrivingCondition.SNOWY: 0.5,
        }
        noise = sensor_noise.get(cond, 0.2)
        
        sensor_data = {
            "camera": np.random.randn(512) * noise,
            "lidar": np.random.randn(256) * (noise * 0.5),
            "radar": np.random.randn(128) * (noise * 0.3),
            "gps": np.random.randn(64) * 0.05,
        }
        
        # 突发紧急情况
        for step in range(5):
            # Step 3 模拟行人突然出现
            if step == 2 and len(scenario["traffic"]) > 0:
                emergency_traffic = scenario["traffic"] + [
                    TrafficParticipant(99, np.array([100.5, 50.0]), np.array([0.0, -1.0]),
                                      -np.pi/2, "pedestrian", 0.5, 0.9)
                ]
            else:
                emergency_traffic = scenario["traffic"]
            
            result = brain.driving_step(
                sensor_data, scenario["vehicle"], emergency_traffic
            )
            
            cmd = result["command"]
            cog = result["cognition"]
            fw = result["fusion_weights"]
            
            print(f"  Step {step+1}: "
                  f"action={cmd.action.value:18s} | "
                  f"speed={cmd.target_speed:.1f}m/s | "
                  f"steer={cmd.steering:+.2f} | "
                  f"conf={cmd.confidence:.2f} risk={cmd.risk_level:.2f} | "
                  f"TTC={cog['min_ttc']:.1f}s | "
                  f"cam={fw['camera']:.2f} lid={fw['lidar']:.2f} rad={fw['radar']:.2f}")
    
    # 跨条件适应性能
    print(f"\n{'=' * 70}")
    print("  跨条件适应性能汇总")
    print(f"{'=' * 70}")
    
    print(f"\n  传感器融合权重动态调整:")
    for cond in DrivingCondition:
        weights = brain._adaptive_fusion_weights(cond)
        print(f"  {cond.value:15s} → cam={weights['camera']:.2f} "
              f"lidar={weights['lidar']:.2f} radar={weights['radar']:.2f} "
              f"gps={weights['gps']:.2f}")
    
    # 条件间迁移
    print(f"\n  跨条件域适应偏移量:")
    conditions = list(DrivingCondition)
    for i, src in enumerate(conditions[:4]):
        for j, tgt in enumerate(conditions[i+1:i+3]):
            src_percept = np.random.randn(512) * 0.3
            adapted = brain.cross_condition_adapt(src_percept, src, tgt)
            shift = np.linalg.norm(adapted - src_percept)
            print(f"  {src.value:15s} → {tgt.value:15s}: 偏移 = {shift:.3f}")


if __name__ == "__main__":
    demo_autonomous_driving()
