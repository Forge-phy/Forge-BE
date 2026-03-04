"""
Forge Isaac Sim Integration Module (v4.0)

Isaac Lab 기반 시뮬레이션 + DR 병렬 실행 + 신뢰도 분석

v4.0 변경:
- LSTM 제거 → DR 분포 분석으로 대체
- Isaac Sim SDK 직접 호출 → Isaac Lab ManagerBasedEnv
- 환경/시뮬/NLP 파서 → Isaac Lab 기반으로 재구현 예정
"""

from .client import (
    IsaacSimClient,
    IsaacSimConfig,
    get_client,
    initialize_isaac_sim,
)

from .schemas import (
    EnvironmentConfig,
    WarehouseConfig,
    RobotConfig,
    SimulationConfig,
    SimulationResult,
    RobotType,
    Position,
    Rotation,
)

__all__ = [
    # Client
    "IsaacSimClient",
    "IsaacSimConfig",
    "get_client",
    "initialize_isaac_sim",
    # Schemas
    "EnvironmentConfig",
    "WarehouseConfig",
    "RobotConfig",
    "SimulationConfig",
    "SimulationResult",
    "RobotType",
    "Position",
    "Rotation",
]
