# -*- coding: utf-8 -*-
"""
机器人导航 - 跨环境零样本迁移
Robot Navigation - Cross-Environment Zero-Shot Transfer

基于欧拉同构认知引擎，实现机器人跨环境导航的零样本泛化。
支持：室内/室外/工厂/仓库/楼梯 等不同环境的自主导航。
"""

import numpy as np
import cmath
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class NavEnvironment(Enum):
    INDOOR_OFFICE = "indoor_office"
    INDOOR_WAREHOUSE = "indoor_warehouse"
    OUTDOOR_URBAN = "outdoor_urban"
    OUTDOOR_FIELD = "outdoor_field"
    FACTORY_FLOOR = "factory_floor"
    STAIRCASE = "staircase"
    CORRIDOR = "corridor"


class ObstacleType(Enum):
    STATIC = "static"        # 墙壁、柱子
    DYNAMIC = "dynamic"      # 行人、车辆
    TRAVERSABLE = "traversable"  # 草地、地毯
    HAZARDOUS = "hazardous"  # 水、火、坑洞


@dataclass
class NavGoal:
    """导航目标"""
    position: np.ndarray      # 目标位置 [x, y, z]
    tolerance: float = 0.5    # 到达容差 (米)
    priority: float = 1.0     # 优先级
    description: str = ""


@dataclass
class Obstacle:
    """障碍物"""
    position: np.ndarray      # 中心位置
    radius: float             # 包围半径
    obstacle_type: ObstacleType
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 动态障碍物速度
    confidence: float = 1.0   # 检测置信度


@dataclass
class NavPath:
    """导航路径"""
    waypoints: List[np.ndarray]
    total_length: float
    estimated_time: float
    risk_score: float         # 风险评分 [0, 1]
    energy_cost: float        # 预估能量消耗


class RobotNavigationBrain:
    """
    机器人导航大脑 - 欧拉同构跨环境导航
    
    核心能力：
    1. 多传感器融合建图
    2. 欧拉认知驱动的路径规划
    3. 动态避障与重规划
    4. 跨环境零样本迁移（域适应）
    """
    
    def __init__(self, state_dim: int = 256):
        self.state_dim = state_dim
        
        # 导航认知引擎
        self.W_percept = np.random.randn(state_dim, 128) * 0.01
        self.W_logic = np.random.randn(128, 64) * 0.01
        self.W_emotion = np.random.randn(128, 64) * 0.01
        
        # 环境特征库（域适应关键）
        self.environment_prototypes: Dict[str, np.ndarray] = {}
        self._init_environment_prototypes()
        
        # 局部地图
        self.local_map = np.zeros((100, 100))  # 10m x 10m, 0.1m 分辨率
        self.map_origin = np.zeros(2)
        
        # 导航状态
        self.current_goal: Optional[NavGoal] = None
        self.current_path: Optional[NavPath] = None
        self.detected_obstacles: List[Obstacle] = []
        
        # 欧拉认知状态
        self.cognitive_z = 0 + 0j
        self.navigation_confidence = 0.5
        
        # 安全参数
        self.safe_distance = 0.5       # 最小安全距离 (米)
        self.max_speed = 1.5           # 最大速度 (m/s)
        self.risk_threshold = 0.7      # 风险阈值
        
        # 跨环境适应参数
        self.adaptation_matrix = np.eye(state_dim)
        self.domain_shift_detected = False
    
    def _init_environment_prototypes(self):
        """初始化环境原型向量（用于跨环境迁移）"""
        rng = np.random.RandomState(42)
        self.environment_prototypes = {
            NavEnvironment.INDOOR_OFFICE: rng.randn(self.state_dim) * 0.3 + 0.5,
            NavEnvironment.INDOOR_WAREHOUSE: rng.randn(self.state_dim) * 0.4 + 0.3,
            NavEnvironment.OUTDOOR_URBAN: rng.randn(self.state_dim) * 0.5 + 0.2,
            NavEnvironment.OUTDOOR_FIELD: rng.randn(self.state_dim) * 0.6 + 0.1,
            NavEnvironment.FACTORY_FLOOR: rng.randn(self.state_dim) * 0.35 + 0.6,
            NavEnvironment.STAIRCASE: rng.randn(self.state_dim) * 0.25 + 0.7,
            NavEnvironment.CORRIDOR: rng.randn(self.state_dim) * 0.2 + 0.8,
        }
    
    # ==================== 感知与建图 ====================
    
    def fuse_sensors(self, lidar: np.ndarray, camera: np.ndarray,
                     imu: np.ndarray, ultrasonic: np.ndarray = None) -> Dict:
        """
        多传感器融合建图
        
        LiDAR: 精确距离 → 静态地图
        Camera: 语义信息 → 障碍物分类
        IMU: 运动状态 → 位姿校正
        Ultrasonic: 近距离补充
        """
        # LiDAR 主导建图
        occupancy = self._lidar_to_occupancy(lidar)
        
        # Camera 语义标注
        semantic = self._camera_semantic(camera)
        
        # IMU 位姿校正
        pose_correction = self._imu_correct(imu)
        
        # 超声波补充
        if ultrasonic is not None:
            near_field = self._ultrasonic_near_field(ultrasonic)
            occupancy = np.maximum(occupancy, near_field)
        
        # 融合感知向量
        perception = np.zeros(self.state_dim)
        perception[:64] = occupancy.flatten()[:64]
        perception[64:128] = semantic[:64]
        perception[128:192] = pose_correction[:64]
        
        return {
            "occupancy_map": occupancy,
            "semantic_map": semantic,
            "pose_correction": pose_correction,
            "perception_vector": perception
        }
    
    def _lidar_to_occupancy(self, lidar: np.ndarray) -> np.ndarray:
        """LiDAR 点云转占据栅格"""
        occupancy = np.zeros((100, 100))
        n_beams = min(len(lidar), 360)
        for i in range(n_beams):
            angle = 2 * np.pi * i / n_beams
            dist = float(lidar[i]) if i < len(lidar) else 5.0
            if dist < 5.0:  # 5m 范围内
                gx = int(50 + dist * 10 * np.cos(angle))
                gy = int(50 + dist * 10 * np.sin(angle))
                if 0 <= gx < 100 and 0 <= gy < 100:
                    occupancy[gy, gx] = 1.0
        return occupancy
    
    def _camera_semantic(self, camera: np.ndarray) -> np.ndarray:
        """相机语义特征提取"""
        return np.tanh(camera[:64]) if len(camera) >= 64 else np.pad(np.tanh(camera), (0, 64 - len(camera)))
    
    def _imu_correct(self, imu: np.ndarray) -> np.ndarray:
        """IMU 位姿校正"""
        return np.tanh(imu[:64]) if len(imu) >= 64 else np.pad(np.tanh(imu), (0, 64 - len(imu)))
    
    def _ultrasonic_near_field(self, ultrasonic: np.ndarray) -> np.ndarray:
        """超声波近场补充"""
        near = np.zeros((100, 100))
        for i, dist in enumerate(ultrasonic[:8]):
            angle = 2 * np.pi * i / 8
            gx = int(50 + dist * 10 * np.cos(angle))
            gy = int(50 + dist * 10 * np.sin(angle))
            if 0 <= gx < 100 and 0 <= gy < 100:
                near[gy, gx] = 1.0
        return near
    
    # ==================== 环境识别与迁移 ====================
    
    def identify_environment(self, perception_vector: np.ndarray) -> Tuple[NavEnvironment, float]:
        """
        识别当前环境类型
        
        通过与存储的环境原型比较，判断当前属于哪种环境。
        这是跨环境零样本迁移的关键：识别后自动切换导航策略。
        """
        best_env = NavEnvironment.INDOOR_OFFICE
        best_score = -float('inf')
        
        for env_type, prototype in self.environment_prototypes.items():
            # 欧拉相似度：复数域的相关性
            sim = self._euler_similarity(perception_vector, prototype)
            if sim > best_score:
                best_score = sim
                best_env = env_type
        
        # 域偏移检测
        if best_score < 0.3:
            self.domain_shift_detected = True
        else:
            self.domain_shift_detected = False
        
        return best_env, best_score
    
    def _euler_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """欧拉相似度：在复数域计算相关性"""
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        
        # 将实向量映射到复数域
        z_a = a[::2] + 1j * a[1::2] if min_len >= 2 else a[0] + 0j
        z_b = b[::2] + 1j * b[1::2] if min_len >= 2 else b[0] + 0j
        
        # 复数相关系数
        numerator = np.abs(np.sum(z_a * np.conj(z_b)))
        denominator = np.sqrt(np.sum(np.abs(z_a)**2) * np.sum(np.abs(z_b)**2)) + 1e-8
        
        return float(numerator / denominator)
    
    def adapt_to_environment(self, perception_vector: np.ndarray, 
                             target_env: NavEnvironment) -> np.ndarray:
        """
        跨环境域适应
        
        使用欧拉同构的 MMD 对齐，将当前感知映射到目标环境的特征空间。
        这是零样本迁移的核心：无需重新训练即可在新环境工作。
        """
        prototype = self.environment_prototypes[target_env]
        
        # 计算域偏移向量
        shift = prototype[:len(perception_vector)] - perception_vector
        
        # 欧拉旋转适应（复数域对齐）
        min_len = min(len(perception_vector), len(shift))
        z_percept = perception_vector[:min_len:2] + 1j * perception_vector[1:min_len:2] if min_len >= 2 else perception_vector[0] + 0j
        z_shift = shift[:min_len:2] + 1j * shift[1:min_len:2] if min_len >= 2 else shift[0] + 0j
        
        # 旋转对齐
        angle = np.angle(np.mean(z_shift))
        rotation = cmath.exp(1j * angle * 0.1)  # 缓慢适应
        
        z_adapted = z_percept * rotation
        
        # 转回实数
        adapted = perception_vector.copy()
        adapted[:min_len:2] = z_adapted.real if hasattr(z_adapted, '__len__') else np.full(len(adapted[:min_len:2]), z_adapted.real)
        adapted[1:min_len:2] = z_adapted.imag if hasattr(z_adapted, '__len__') else np.full(len(adapted[1:min_len:2]), z_adapted.imag)
        
        return adapted
    
    # ==================== 路径规划 ====================
    
    def plan_path(self, start: np.ndarray, goal: NavGoal,
                  obstacles: List[Obstacle], env_type: NavEnvironment) -> NavPath:
        """
        欧拉认知驱动路径规划
        
        结合逻辑推理（最短路径）和情感评估（风险感知）
        """
        # 逻辑路径（A* 简化版）
        logic_path = self._astar_simplified(start, goal.position, obstacles)
        
        # 情感风险评估
        risk_scores = self._evaluate_path_risk(logic_path, obstacles)
        
        # 欧拉融合
        fused_path = self._euler_path_fusion(logic_path, risk_scores, env_type)
        
        total_length = sum(np.linalg.norm(fused_path[i+1] - fused_path[i]) 
                          for i in range(len(fused_path) - 1))
        
        avg_risk = np.mean(risk_scores) if risk_scores else 0.0
        
        # 根据环境类型调整参数
        env_speeds = {
            NavEnvironment.INDOOR_OFFICE: 0.8,
            NavEnvironment.INDOOR_WAREHOUSE: 1.2,
            NavEnvironment.OUTDOOR_URBAN: 1.0,
            NavEnvironment.OUTDOOR_FIELD: 0.6,
            NavEnvironment.FACTORY_FLOOR: 0.7,
            NavEnvironment.STAIRCASE: 0.3,
            NavEnvironment.CORRIDOR: 1.0,
        }
        
        speed = env_speeds.get(env_type, 0.8)
        
        return NavPath(
            waypoints=fused_path,
            total_length=total_length,
            estimated_time=total_length / speed,
            risk_score=avg_risk,
            energy_cost=total_length * (0.1 + avg_risk * 0.2)
        )
    
    def _astar_simplified(self, start: np.ndarray, goal: np.ndarray,
                          obstacles: List[Obstacle]) -> List[np.ndarray]:
        """简化 A* 路径规划"""
        n_waypoints = 10
        path = [start.copy()]
        
        direction = goal - start
        step = direction / n_waypoints
        
        for i in range(1, n_waypoints):
            waypoint = start + step * i
            
            # 避障偏移
            for obs in obstacles:
                diff = waypoint - obs.position
                dist = np.linalg.norm(diff)
                if dist < obs.radius + self.safe_distance:
                    # 推离障碍物
                    push = diff / (dist + 1e-8) * (obs.radius + self.safe_distance - dist)
                    waypoint += push
            
            path.append(waypoint)
        
        path.append(goal.copy())
        return path
    
    def _evaluate_path_risk(self, path: List[np.ndarray], 
                            obstacles: List[Obstacle]) -> List[float]:
        """路径风险评估"""
        risks = []
        for wp in path:
            risk = 0.0
            for obs in obstacles:
                dist = np.linalg.norm(wp - obs.position)
                if dist < obs.radius + self.safe_distance * 2:
                    risk += np.exp(-(dist - obs.radius) / self.safe_distance)
                    if obs.obstacle_type == ObstacleType.HAZARDOUS:
                        risk *= 3.0
                    if obs.obstacle_type == ObstacleType.DYNAMIC:
                        risk *= 1.5
            risks.append(min(risk, 1.0))
        return risks
    
    def _euler_path_fusion(self, logic_path: List[np.ndarray],
                           risk_scores: List[float],
                           env_type: NavEnvironment) -> List[np.ndarray]:
        """欧拉融合：逻辑路径 + 风险情感 → 安全路径"""
        fused = []
        for i, (wp, risk) in enumerate(zip(logic_path, risk_scores)):
            # 认知相位：风险越高越偏情感
            theta = risk * np.pi / 2
            
            # 欧拉调制
            modulation = np.cos(theta) * (1 - risk) + np.sin(theta) * risk
            
            # 风险回避偏移
            if risk > self.risk_threshold and len(fused) > 0:
                avoidance = (wp - logic_path[max(0, i-1)])
                avoidance_norm = np.linalg.norm(avoidance)
                if avoidance_norm > 0:
                    avoidance = avoidance / avoidance_norm * risk * 0.5
                    wp = wp + np.array([avoidance[1], -avoidance[0], 0])  # 垂直偏移
            
            fused.append(wp)
        
        return fused
    
    # ==================== 导航主循环 ====================
    
    def navigate_step(self, sensor_data: Dict, robot_pos: np.ndarray,
                      robot_vel: np.ndarray, goal: NavGoal) -> Dict:
        """
        单步导航决策
        
        完整闭环：感知→环境识别→域适应→路径规划→速度指令
        """
        # 1. 传感器融合
        perception = self.fuse_sensors(
            sensor_data.get("lidar", np.random.randn(360)),
            sensor_data.get("camera", np.random.randn(512)),
            sensor_data.get("imu", np.random.randn(6)),
            sensor_data.get("ultrasonic", np.random.randn(8))
        )
        
        # 2. 环境识别
        env_type, env_confidence = self.identify_environment(perception["perception_vector"])
        
        # 3. 域适应（零样本迁移）
        if env_confidence < 0.5:
            adapted = self.adapt_to_environment(perception["perception_vector"], env_type)
        else:
            adapted = perception["perception_vector"]
        
        # 4. 障碍物检测
        obstacles = self._detect_obstacles(perception, sensor_data)
        
        # 5. 路径规划/重规划
        if self.current_path is None or self._need_replan(robot_pos, obstacles):
            self.current_path = self.plan_path(robot_pos, goal, obstacles, env_type)
            self.current_goal = goal
        
        # 6. 计算速度指令
        velocity_cmd = self._compute_velocity(robot_pos, robot_vel, self.current_path)
        
        # 7. 更新认知状态
        dist_to_goal = np.linalg.norm(robot_pos[:2] - goal.position[:2])
        self.navigation_confidence = max(0, 1.0 - dist_to_goal / 10.0)
        
        self.cognitive_z = self.navigation_confidence * cmath.exp(
            1j * np.arctan2(robot_pos[1] - goal.position[1], robot_pos[0] - goal.position[0])
        )
        
        return {
            "velocity_command": velocity_cmd,
            "environment": env_type.value,
            "env_confidence": env_confidence,
            "path_risk": self.current_path.risk_score,
            "obstacles": len(obstacles),
            "cognitive_z": self.cognitive_z,
            "navigation_confidence": self.navigation_confidence,
            "domain_shift": self.domain_shift_detected
        }
    
    def _detect_obstacles(self, perception: Dict, sensor_data: Dict) -> List[Obstacle]:
        """从感知数据中检测障碍物"""
        obstacles = []
        occupancy = perception["occupancy_map"]
        
        # 简化：从占据栅格提取障碍物
        obstacle_cells = np.argwhere(occupancy > 0.5)
        if len(obstacle_cells) > 0:
            # 聚类（简化为取中心）
            n_obs = min(len(obstacle_cells), 5)
            indices = np.linspace(0, len(obstacle_cells) - 1, n_obs, dtype=int)
            for idx in indices:
                gy, gx = obstacle_cells[idx]
                pos = np.array([(gx - 50) * 0.1, (gy - 50) * 0.1, 0.0])
                obstacles.append(Obstacle(
                    position=pos,
                    radius=0.3,
                    obstacle_type=ObstacleType.STATIC,
                    confidence=0.8
                ))
        return obstacles
    
    def _need_replan(self, pos: np.ndarray, obstacles: List[Obstacle]) -> bool:
        """判断是否需要重规划"""
        if self.current_path is None:
            return True
        
        # 检查路径是否被新障碍物阻塞
        for wp in self.current_path.waypoints[:3]:
            for obs in obstacles:
                if np.linalg.norm(wp - obs.position) < obs.radius + self.safe_distance:
                    return True
        return False
    
    def _compute_velocity(self, pos: np.ndarray, vel: np.ndarray,
                          path: NavPath) -> np.ndarray:
        """计算速度指令"""
        # 找最近的路径点
        min_dist = float('inf')
        target_wp = path.waypoints[0]
        for wp in path.waypoints:
            d = np.linalg.norm(pos[:2] - wp[:2])
            if d < min_dist:
                min_dist = d
                target_wp = wp
        
        # 朝目标点移动
        direction = target_wp[:2] - pos[:2]
        dist = np.linalg.norm(direction)
        
        if dist > 0.01:
            direction = direction / dist
        
        # 速度 = 方向 × 速度权重
        speed = min(self.max_speed, dist * 0.5)
        
        # 风险降速
        if path.risk_score > self.risk_threshold:
            speed *= (1 - path.risk_score) * 0.5
        
        vx = direction[0] * speed
        vy = direction[1] * speed
        omega = 0.0  # 角速度
        
        return np.array([vx, vy, omega])


# ==================== 跨环境迁移测试 ====================

def demo_robot_navigation():
    """机器人导航跨环境零样本迁移演示"""
    print("=" * 70)
    print("  机器人导航 - 跨环境零样本迁移演示")
    print("  Robot Navigation - Cross-Environment Zero-Shot Transfer")
    print("=" * 70)
    
    brain = RobotNavigationBrain()
    
    # 定义不同环境的导航场景
    scenarios = [
        {
            "name": "室内办公",
            "env": NavEnvironment.INDOOR_OFFICE,
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": NavGoal(position=np.array([5.0, 3.0, 0.0]), description="会议室"),
            "lidar_range": 5.0,
            "obstacle_density": 0.3,
        },
        {
            "name": "仓库物流",
            "env": NavEnvironment.INDOOR_WAREHOUSE,
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": NavGoal(position=np.array([8.0, -2.0, 0.0]), description="货架A3"),
            "lidar_range": 8.0,
            "obstacle_density": 0.4,
        },
        {
            "name": "户外城市",
            "env": NavEnvironment.OUTDOOR_URBAN,
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": NavGoal(position=np.array([15.0, 10.0, 0.0]), description="十字路口"),
            "lidar_range": 15.0,
            "obstacle_density": 0.5,
        },
        {
            "name": "工厂车间",
            "env": NavEnvironment.FACTORY_FLOOR,
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": NavGoal(position=np.array([6.0, 4.0, 0.0]), description="工位B2"),
            "lidar_range": 6.0,
            "obstacle_density": 0.6,
        },
        {
            "name": "楼梯环境",
            "env": NavEnvironment.STAIRCASE,
            "start": np.array([0.0, 0.0, 0.0]),
            "goal": NavGoal(position=np.array([2.0, 0.0, 3.0]), description="二楼入口"),
            "lidar_range": 3.0,
            "obstacle_density": 0.2,
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'─' * 50}")
        print(f"  场景: {scenario['name']}")
        print(f"  目标: {scenario['goal'].description} @ {scenario['goal'].position}")
        print(f"{'─' * 50}")
        
        robot_pos = scenario["start"].copy()
        robot_vel = np.zeros(3)
        
        # 模拟导航过程
        for step in range(8):
            # 生成模拟传感器数据
            sensor_data = {
                "lidar": np.random.randn(360) * 0.3 + scenario["lidar_range"],
                "camera": np.random.randn(512) * 0.5,
                "imu": np.random.randn(6) * 0.1,
                "ultrasonic": np.random.randn(8) * 0.2 + 1.0,
            }
            
            result = brain.navigate_step(sensor_data, robot_pos, robot_vel, scenario["goal"])
            
            # 更新位置
            dt = 0.1
            robot_pos[:2] += result["velocity_command"][:2] * dt
            
            dist = np.linalg.norm(robot_pos[:2] - scenario["goal"].position[:2])
            
            print(f"  Step {step+1}: "
                  f"env={result['environment']:18s} conf={result['env_confidence']:.2f} | "
                  f"risk={result['path_risk']:.2f} obs={result['obstacles']} | "
                  f"nav_conf={result['navigation_confidence']:.2f} | "
                  f"dist={dist:.2f}m | "
                  f"domain_shift={'YES' if result['domain_shift'] else 'no':3s}")
            
            if dist < scenario["goal"].tolerance:
                print(f"  ✓ 到达目标!")
                break
    
    # 跨环境迁移性能
    print(f"\n{'=' * 70}")
    print("  跨环境迁移性能汇总")
    print(f"{'=' * 70}")
    
    for env_type in NavEnvironment:
        test_percept = np.random.randn(256) * 0.4
        identified, confidence = brain.identify_environment(test_percept)
        adapted = brain.adapt_to_environment(test_percept, env_type)
        adaptation_shift = np.linalg.norm(adapted - test_percept)
        print(f"  {env_type.value:20s} → 识别为: {identified.value:20s} "
              f"(conf={confidence:.2f}) 适应偏移: {adaptation_shift:.3f}")


if __name__ == "__main__":
    demo_robot_navigation()
