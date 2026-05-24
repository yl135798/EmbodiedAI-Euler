"""
决策执行层 - 资源分配器 + 元权力建模
实现"行"：钱（时间＋五感信息）权（元权力，定义权）分配
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class ResourceCost:
    time_cost: float       # 时间成本（秒）
    info_cost: float       # 五感信息消耗（比特）
    compute_cost: float    # 算力成本（TFLOPS）
    money_cost: float      # 金钱成本（元）


class ResourceAllocator:
    """
    资源分配器
    对"钱（时间+五感信息）"进行量化分配
    """

    def __init__(self, total_budget: Dict[str, float] = None):
        # 默认资源预算
        self.total_budget = total_budget or {
            "time": 100.0,      # 秒
            "info": 10000.0,    # 比特
            "compute": 10.0,    # TFLOPS
            "money": 100.0,     # 元
        }
        self.remaining = dict(self.total_budget)

    def allocate(self,
                task_vector: np.ndarray,
                task_requirements: ResourceCost,
                urgency: float = 0.5) -> Dict:
        """
        根据任务向量和需求分配资源
        返回分配方案 + 是否可行
        """
        # 检查预算是否足够
        feasible = all([
            self.remaining["time"] >= task_requirements.time_cost,
            self.remaining["info"] >= task_requirements.info_cost,
            self.remaining["compute"] >= task_requirements.compute_cost,
            self.remaining["money"] >= task_requirements.money_cost,
        ])

        if not feasible:
            # 按比例缩减
            scale = self._compute_feasible_scale(task_requirements)
        else:
            scale = 1.0

        allocated = ResourceCost(
            time_cost=task_requirements.time_cost * scale,
            info_cost=task_requirements.info_cost * scale,
            compute_cost=task_requirements.compute_cost * scale,
            money_cost=task_requirements.money_cost * scale,
        )

        # 紧急任务优先分配
        if urgency > 0.7:
            allocated.time_cost *= 0.8  # 紧急任务时间打折（加急）

        # 扣除资源
        self.remaining["time"] -= allocated.time_cost
        self.remaining["info"] -= allocated.info_cost
        self.remaining["compute"] -= allocated.compute_cost
        self.remaining["money"] -= allocated.money_cost

        return {
            "allocated": allocated,
            "feasible": feasible,
            "scale": scale,
            "remaining": dict(self.remaining),
        }

    def _compute_feasible_scale(self, req: ResourceCost) -> float:
        scales = []
        for key, cost in zip(
            ["time", "info", "compute", "money"],
            [req.time_cost, req.info_cost, req.compute_cost, req.money_cost]
        ):
            if cost > 0:
                scales.append(self.remaining[key] / cost)
        return max(0.0, min(scales)) if scales else 0.0

    def reset(self):
        self.remaining = dict(self.total_budget)


class MetaPowerAllocator:
    """
    元权力 / 定义权分配器
    模拟"谁有权定义问题" —— 这是最高层的权力

    权力维度：
    - agenda_setting: 议程设置权（定义要解决的问题）
    - framing: 框架定义权（定义问题的解释方式）
    - metric_definition: 指标定义权（定义什么叫"好"）
    - resource_control: 资源控制权
    """

    POWER_DIMENSIONS = [
        "agenda_setting",
        "framing",
        "metric_definition",
        "resource_control",
    ]

    def __init__(self):
        # 权力分布矩阵：每个 agent/模块 的权力向量
        self.power_distribution: Dict[str, np.ndarray] = {}
        self.definition_history: List[Dict] = []

    def register_agent(self, agent_id: str, init_power: np.ndarray = None):
        """注册一个 agent 的权力向量"""
        if init_power is None:
            init_power = np.ones(len(self.POWER_DIMENSIONS)) / len(self.POWER_DIMENSIONS)
        self.power_distribution[agent_id] = init_power

    def allocate_power(self,
                     task_context: Dict,
                     claimants: List[str]) -> Dict[str, float]:
        """
        针对特定任务，分配各 claimant 的元权力权重
        返回: {agent_id: power_weight}
        """
        weights = {}
        for agent_id in claimants:
            if agent_id not in self.power_distribution:
                self.register_agent(agent_id)

            power_vec = self.power_distribution[agent_id]

            # 根据任务上下文调整权力
            contextual_boost = self._contextual_power_boost(agent_id, task_context)
            weight = float(np.sum(power_vec) * contextual_boost)
            weights[agent_id] = weight

        # 归一化
        total = sum(weights.values()) + 1e-8
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def _contextual_power_boost(self, agent_id: str, context: Dict) -> float:
        """根据上下文调整权力权重"""
        boost = 1.0
        # 历史表现好的 agent 获得权力加成
        if agent_id in context.get("high_performers", []):
            boost *= 1.5
        # 专业匹配度
        if agent_id in context.get("domain_experts", []):
            boost *= 1.3
        return boost

    def update_power(self, agent_id: str, performance_feedback: float):
        """
        根据表现反馈更新权力分布
        performance_feedback: [-1, 1]，正=表现好，负=表现差
        """
        if agent_id not in self.power_distribution:
            return
        # 权力根据表现奖惩（类似学习率）
        lr = 0.1
        self.power_distribution[agent_id] += lr * performance_feedback
        self.power_distribution[agent_id] = np.clip(
            self.power_distribution[agent_id], 0.01, 1.0
        )
        # 归一化
        self.power_distribution[agent_id] /= (
            np.sum(self.power_distribution[agent_id]) + 1e-8
        )


class Executor:
    """
    执行引擎：将"知"的结果转化为"行"的量化输出
    核心公式：知（逻辑+情绪）+ 资源分配 + 权力分配 → 量化结果
    """

    def __init__(self, state_dim: int = 512):
        self.state_dim = state_dim
        self.resource_allocator = ResourceAllocator()
        self.power_allocator = MetaPowerAllocator()
        self.execution_log: List[Dict] = []

    def execute(self,
                blended_cognition: np.ndarray,
                task_context: Dict,
                claimants: List[str]) -> Dict:
        """
        完整执行流程
        返回量化结果
        """
        # 1. 资源分配
        dummy_cost = ResourceCost(
            time_cost=10.0,
            info_cost=100.0,
            compute_cost=1.0,
            money_cost=5.0,
        )
        resource_result = self.resource_allocator.allocate(
            blended_cognition, dummy_cost,
            urgency=task_context.get("urgency", 0.5)
        )

        # 2. 权力分配
        power_weights = self.power_allocator.allocate_power(task_context, claimants)

        # 3. 量化输出（知行合一）
        result = self._quantize(blended_cognition, resource_result, power_weights)

        # 4. 记录
        log_entry = {
            "timestamp": time.time(),
            "cognition_norm": float(np.linalg.norm(blended_cognition)),
            "resource_feasible": resource_result["feasible"],
            "power_weights": power_weights,
            "quantized_result": result,
        }
        self.execution_log.append(log_entry)

        return result

    def _quantize(self,
                  cognition: np.ndarray,
                  resource_result: Dict,
                  power_weights: Dict[str, float]) -> Dict:
        """
        知行合一 → 量化结果
        将认知向量、资源分配、权力分配融合为量化输出
        """
        # 认知向量主导输出
        base_output = cognition / (np.linalg.norm(cognition) + 1e-8)

        # 资源可行性调制
        resource_factor = 1.0 if resource_result["feasible"] else resource_result["scale"]

        # 权力集中度（Herfindahl指数）：权力越集中，输出越确定
        weights = np.array(list(power_weights.values()))
        power_concentration = float(np.sum(weights ** 2))

        # 最终量化分数
        quantized_score = float(
            np.mean(np.abs(base_output)) * resource_factor * (1 + power_concentration)
        )

        return {
            "quantized_score": quantized_score,
            "resource_factor": resource_factor,
            "power_concentration": power_concentration,
            "power_weights": power_weights,
            "output_vector": base_output[:16].tolist(),  # 取前16维用于展示
            "interpretation": self._interpret_score(quantized_score),
        }

    def _interpret_score(self, score: float) -> str:
        if score > 0.8:
            return "高置信度执行决策"
        elif score > 0.5:
            return "中等置信度，建议谨慎执行"
        elif score > 0.3:
            return "低置信度，需进一步感知"
        else:
            return "无法决策，感知不足"
