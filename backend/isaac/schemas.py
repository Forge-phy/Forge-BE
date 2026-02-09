"""
Forge Isaac Sim - 데이터 스키마
시뮬레이션 입출력 데이터 타입 정의
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RobotType(Enum):
    """로봇 타입"""
    AGV = "agv"              # 무인 운반차
    AMR = "amr"              # 자율 이동 로봇
    FORKLIFT = "forklift"    # 지게차
    ARM = "arm"              # 로봇팔


class EnvironmentType(Enum):
    """환경 타입"""
    WAREHOUSE = "warehouse"   # 물류 창고
    FACTORY = "factory"       # 공장
    CUSTOM = "custom"         # 커스텀


@dataclass
class Position:
    """3D 위치"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Rotation:
    """3D 회전 (euler angles in degrees)"""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class RobotConfig:
    """로봇 설정"""
    robot_id: str
    robot_type: RobotType = RobotType.AGV
    position: Position = field(default_factory=Position)
    rotation: Rotation = field(default_factory=Rotation)
    speed: float = 1.0  # m/s
    payload_capacity: float = 100.0  # kg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type.value,
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "rotation": {"roll": self.rotation.roll, "pitch": self.rotation.pitch, "yaw": self.rotation.yaw},
            "speed": self.speed,
            "payload_capacity": self.payload_capacity
        }


@dataclass
class WarehouseConfig:
    """창고 환경 설정"""
    width: float = 30.0   # m
    length: float = 20.0  # m
    height: float = 5.0   # m
    shelf_rows: int = 2
    shelf_columns: int = 5
    aisle_width: float = 3.0  # m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "shelf_rows": self.shelf_rows,
            "shelf_columns": self.shelf_columns,
            "aisle_width": self.aisle_width
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarehouseConfig":
        return cls(
            width=data.get("width", 30.0),
            length=data.get("length", 20.0),
            height=data.get("height", 5.0),
            shelf_rows=data.get("shelf_rows", 2),
            shelf_columns=data.get("shelf_columns", 5),
            aisle_width=data.get("aisle_width", 3.0)
        )


@dataclass
class EnvironmentConfig:
    """시뮬레이션 환경 설정"""
    env_type: EnvironmentType = EnvironmentType.WAREHOUSE
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    robots: List[RobotConfig] = field(default_factory=list)
    temperature: float = 25.0  # °C
    humidity: float = 50.0     # %

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_type": self.env_type.value,
            "warehouse": self.warehouse.to_dict(),
            "robots": [r.to_dict() for r in self.robots],
            "temperature": self.temperature,
            "humidity": self.humidity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentConfig":
        # 환경 타입 파싱
        env_type_str = data.get("env_type", "warehouse")
        env_type = EnvironmentType(env_type_str) if isinstance(env_type_str, str) else EnvironmentType.WAREHOUSE

        # 창고 설정 파싱
        warehouse_data = data.get("warehouse", {})
        warehouse = WarehouseConfig.from_dict(warehouse_data)

        # 로봇 설정 파싱
        robots = []
        robots_data = data.get("robots", [])
        for i, r in enumerate(robots_data):
            if isinstance(r, dict):
                robot_type_str = r.get("robot_type", r.get("type", "agv"))
                try:
                    robot_type = RobotType(robot_type_str.lower())
                except ValueError:
                    robot_type = RobotType.AGV

                count = r.get("count", 1)
                for j in range(count):
                    robots.append(RobotConfig(
                        robot_id=r.get("robot_id", f"robot_{i}_{j}"),
                        robot_type=robot_type,
                        speed=r.get("speed", 1.0),
                        payload_capacity=r.get("payload_capacity", 100.0)
                    ))

        return cls(
            env_type=env_type,
            warehouse=warehouse,
            robots=robots,
            temperature=data.get("temperature", 25.0),
            humidity=data.get("humidity", 50.0)
        )


@dataclass
class SimulationConfig:
    """시뮬레이션 실행 설정"""
    duration: float = 3600.0     # 시뮬레이션 시간 (초)
    time_step: float = 1/60      # 물리 시뮬레이션 스텝 (초)
    realtime_factor: float = 1.0 # 실시간 대비 속도
    record_video: bool = False
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "time_step": self.time_step,
            "realtime_factor": self.realtime_factor,
            "record_video": self.record_video,
            "output_path": self.output_path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        return cls(
            duration=data.get("duration", 3600.0),
            time_step=data.get("time_step", 1/60),
            realtime_factor=data.get("realtime_factor", 1.0),
            record_video=data.get("record_video", False),
            output_path=data.get("output_path")
        )


@dataclass
class RobotMetrics:
    """로봇별 시뮬레이션 결과"""
    robot_id: str
    tasks_completed: int = 0
    distance_traveled: float = 0.0  # m
    items_transported: int = 0
    collisions: int = 0
    idle_time: float = 0.0  # 초
    active_time: float = 0.0  # 초
    average_speed: float = 0.0  # m/s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "tasks_completed": self.tasks_completed,
            "distance_traveled": self.distance_traveled,
            "items_transported": self.items_transported,
            "collisions": self.collisions,
            "idle_time": self.idle_time,
            "active_time": self.active_time,
            "average_speed": self.average_speed
        }


@dataclass
class SimulationResult:
    """시뮬레이션 결과"""
    success: bool = True
    error_message: Optional[str] = None

    # 전체 메트릭
    total_duration: float = 0.0        # 초
    total_throughput: float = 0.0      # 개/시간
    total_collisions: int = 0
    total_items_processed: int = 0

    # 로봇별 메트릭
    robot_metrics: List[RobotMetrics] = field(default_factory=list)

    # 환경 조건
    avg_temperature: float = 25.0
    avg_humidity: float = 50.0

    # 타임스탬프별 데이터 (시계열)
    timeseries_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error_message": self.error_message,
            "total_duration": self.total_duration,
            "total_throughput": self.total_throughput,
            "total_collisions": self.total_collisions,
            "total_items_processed": self.total_items_processed,
            "robot_metrics": [r.to_dict() for r in self.robot_metrics],
            "avg_temperature": self.avg_temperature,
            "avg_humidity": self.avg_humidity,
            "timeseries_data": self.timeseries_data
        }


@dataclass
class NLPRequest:
    """자연어 요청"""
    text: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class NLPResponse:
    """자연어 파싱 결과"""
    success: bool = True
    environment_config: Optional[EnvironmentConfig] = None
    simulation_config: Optional[SimulationConfig] = None
    error_message: Optional[str] = None
    confidence: float = 0.0
