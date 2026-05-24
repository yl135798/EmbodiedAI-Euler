"""
五感感知层 - 多模态传感器数据融合
支持：视觉、听觉、触觉、嗅觉、味觉
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class SensorReading:
    modality: str          # 感官模态: visual/auditory/tactile/olfactory/gustatory
    timestamp: float
    data: np.ndarray       # 原始传感器数据
    confidence: float      # 置信度 [0, 1]
    metadata: Dict         # 附加信息（传感器ID、位置等）


class FiveSensePerception:
    """
    五感感知融合引擎
    将多模态传感器输入统一编码为向量表示
    """

    MODALITY_DIMS = {
        "visual": 512,
        "auditory": 256,
        "tactile": 128,
        "olfactory": 64,
        "gustatory": 64,
    }

    def __init__(self, fusion_dim: int = 512):
        self.fusion_dim = fusion_dim
        self.readings: List[SensorReading] = []
        self.projection_matrices = self._init_projections()

    def _init_projections(self) -> Dict[str, np.ndarray]:
        """初始化各模态到统一空间的投影矩阵"""
        np.random.seed(42)
        projections = {}
        for mod, dim in self.MODALITY_DIMS.items():
            projections[mod] = np.random.randn(dim, self.fusion_dim) * 0.1
        return projections

    def ingest(self, reading: SensorReading) -> None:
        """摄入一个感官读数"""
        self.readings.append(reading)

    def fuse(self, window_sec: float = 1.0) -> np.ndarray:
        """
        融合时间窗口内的所有感官输入
        返回: (fusion_dim,) 的融合向量
        """
        now = time.time()
        window_readings = [
            r for r in self.readings
            if now - r.timestamp <= window_sec
        ]

        if not window_readings:
            return np.zeros(self.fusion_dim)

        fused = np.zeros(self.fusion_dim)
        total_weight = 0.0

        for r in window_readings:
            proj = self.projection_matrices[r.modality]
            vec = r.data[:self.MODALITY_DIMS[r.modality]]
            projected = vec @ proj / (np.linalg.norm(vec) + 1e-8)
            weight = r.confidence
            fused += weight * projected
            total_weight += weight

        if total_weight > 0:
            fused /= total_weight

        return fused

    def get_modality_status(self) -> Dict[str, bool]:
        """返回各感官模态是否在线"""
        now = time.time()
        status = {mod: False for mod in self.MODALITY_DIMS}
        for r in self.readings:
            if now - r.timestamp <= 5.0:
                status[r.modality] = True
        return status
