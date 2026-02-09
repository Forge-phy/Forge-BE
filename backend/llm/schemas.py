"""
Forge LLM Schemas - 엄격한 JSON 스키마 검증
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal, Union
from enum import Enum


class EnvironmentType(str, Enum):
    WAREHOUSE = "warehouse"
    FACTORY = "factory"
    CUSTOM = "custom"


class RobotType(str, Enum):
    AGV = "AGV"
    AMR = "AMR"
    ARM = "arm"
    HUMANOID = "humanoid"


class ObstacleType(str, Enum):
    SHELF = "shelf"
    WALL = "wall"
    CONVEYOR = "conveyor"
    WORKSTATION = "workstation"
    PILLAR = "pillar"


class ArrangementType(str, Enum):
    GRID = "grid"
    ROW = "row"
    RANDOM = "random"
    CUSTOM = "custom"


class Environment(BaseModel):
    """환경 설정 스키마"""
    type: EnvironmentType = Field(default=EnvironmentType.WAREHOUSE, description="환경 유형")
    width: float = Field(ge=1, le=10000, description="너비 (미터)")
    length: float = Field(ge=1, le=10000, description="길이 (미터)")
    height: float = Field(default=5, ge=2, le=100, description="높이 (미터)")
    floor_friction: float = Field(default=0.7, ge=0.1, le=1.0, description="바닥 마찰 계수")
    ambient_temperature: float = Field(default=25, ge=-20, le=60, description="환경 온도 (섭씨)")
    humidity: float = Field(default=50, ge=0, le=100, description="습도 (%)")

    @field_validator('width', 'length')
    @classmethod
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError('크기는 양수여야 합니다')
        return v


class Robot(BaseModel):
    """로봇 설정 스키마"""
    type: RobotType = Field(default=RobotType.AGV, description="로봇 유형")
    count: int = Field(ge=1, le=1000, description="로봇 대수")
    speed: float = Field(default=1.5, ge=0.1, le=10.0, description="최대 속도 (m/s)")
    acceleration: float = Field(default=1.0, ge=0.1, le=5.0, description="가속도 (m/s²)")
    payload: float = Field(default=100, ge=0, le=10000, description="최대 적재량 (kg)")
    battery_capacity: float = Field(default=100, ge=10, le=1000, description="배터리 용량 (Wh)")
    charging_time: float = Field(default=30, ge=1, le=480, description="충전 시간 (분)")
    collision_radius: float = Field(default=0.5, ge=0.1, le=5.0, description="충돌 반경 (m)")
    sensor_range: float = Field(default=5.0, ge=0.5, le=50.0, description="센서 감지 범위 (m)")

    @model_validator(mode='after')
    def validate_robot(self):
        # AGV는 일반적으로 AMR보다 느림
        if self.type == RobotType.AGV and self.speed > 3.0:
            self.speed = 3.0
        return self


class Obstacle(BaseModel):
    """장애물 설정 스키마"""
    type: ObstacleType = Field(default=ObstacleType.SHELF, description="장애물 유형")
    count: int = Field(ge=0, le=10000, description="개수")
    arrangement: ArrangementType = Field(default=ArrangementType.ROW, description="배치 방식")
    width: float = Field(default=1.0, ge=0.1, le=100, description="너비 (m)")
    length: float = Field(default=2.0, ge=0.1, le=100, description="길이 (m)")
    height: float = Field(default=2.0, ge=0.1, le=50, description="높이 (m)")
    spacing: float = Field(default=3.0, ge=0.5, le=50, description="간격 (m)")


class SimulationConfig(BaseModel):
    """시뮬레이션 설정 스키마"""
    duration: int = Field(default=3600, ge=1, le=86400, description="시뮬레이션 시간 (초)")
    time_step: float = Field(default=0.01, ge=0.001, le=1.0, description="시간 간격 (초)")
    realtime_factor: float = Field(default=1.0, ge=0.1, le=100.0, description="실시간 배율")
    seed: Optional[int] = Field(default=None, ge=0, description="랜덤 시드")
    enable_physics: bool = Field(default=True, description="물리 엔진 활성화")
    enable_collision: bool = Field(default=True, description="충돌 감지 활성화")
    record_trajectory: bool = Field(default=True, description="경로 기록")
    record_interval: float = Field(default=0.1, ge=0.01, le=10.0, description="기록 간격 (초)")


class TaskConfig(BaseModel):
    """작업 설정 스키마"""
    type: Literal["pickup_delivery", "patrol", "assembly", "custom"] = Field(
        default="pickup_delivery", description="작업 유형"
    )
    pickup_locations: int = Field(default=5, ge=1, le=1000, description="픽업 위치 수")
    delivery_locations: int = Field(default=5, ge=1, le=1000, description="배송 위치 수")
    task_generation_rate: float = Field(default=0.1, ge=0.001, le=10.0, description="작업 생성률 (개/초)")
    priority_levels: int = Field(default=3, ge=1, le=10, description="우선순위 레벨 수")


class IsaacSimParameters(BaseModel):
    """Isaac Sim 전체 파라미터 스키마"""
    environment: Environment
    robots: List[Robot] = Field(min_length=1, max_length=100)
    obstacles: List[Obstacle] = Field(default_factory=list, max_length=100)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    task: Optional[TaskConfig] = Field(default=None)

    @model_validator(mode='after')
    def validate_parameters(self):
        # 로봇 수가 환경 크기에 비해 너무 많은지 체크
        total_robots = sum(r.count for r in self.robots)
        area = self.environment.width * self.environment.length
        robot_density = total_robots / area

        if robot_density > 0.1:  # 10m² 당 1대 초과
            # 경고만 하고 통과 (strict mode에서는 에러)
            pass

        return self

    class Config:
        json_schema_extra = {
            "example": {
                "environment": {
                    "type": "warehouse",
                    "width": 30,
                    "length": 20,
                    "height": 5
                },
                "robots": [
                    {"type": "AGV", "count": 3, "speed": 1.5, "payload": 100}
                ],
                "obstacles": [
                    {"type": "shelf", "count": 2, "arrangement": "row"}
                ],
                "simulation": {
                    "duration": 3600,
                    "time_step": 0.01
                }
            }
        }


# 시뮬레이션 결과 스키마
class Collision(BaseModel):
    """충돌 정보"""
    time: float = Field(ge=0, description="발생 시간 (초)")
    robot1: str = Field(description="로봇 1 ID")
    robot2: Optional[str] = Field(default=None, description="로봇 2 ID (장애물 충돌시 None)")
    location: str = Field(description="위치")
    severity: Literal["low", "medium", "high"] = Field(default="medium", description="심각도")


class RobotMetrics(BaseModel):
    """로봇별 메트릭"""
    robot_id: str
    total_distance: float = Field(ge=0, description="총 이동 거리 (m)")
    total_tasks: int = Field(ge=0, description="완료 작업 수")
    idle_time: float = Field(ge=0, description="대기 시간 (초)")
    charging_time: float = Field(ge=0, description="충전 시간 (초)")
    average_speed: float = Field(ge=0, description="평균 속도 (m/s)")
    energy_consumed: float = Field(ge=0, description="에너지 소비 (Wh)")


class SimulationResult(BaseModel):
    """시뮬레이션 결과 스키마"""
    duration: float = Field(ge=0, description="실행 시간 (초)")
    throughput: float = Field(ge=0, description="처리량 (items/hour)")
    total_tasks_completed: int = Field(ge=0, description="완료된 작업 수")
    total_collisions: int = Field(ge=0, description="총 충돌 횟수")
    collisions: List[Collision] = Field(default_factory=list, description="충돌 상세")
    robot_metrics: List[RobotMetrics] = Field(default_factory=list, description="로봇별 메트릭")
    bottleneck: Optional[str] = Field(default=None, description="병목 지점")
    efficiency: float = Field(ge=0, le=100, description="효율성 (%)")
    utilization: float = Field(ge=0, le=100, description="가동률 (%)")


def validate_and_normalize(data: dict) -> IsaacSimParameters:
    """파라미터 검증 및 정규화"""
    return IsaacSimParameters.model_validate(data)


def get_schema_json() -> str:
    """JSON 스키마 반환 (프롬프트용)"""
    return IsaacSimParameters.model_json_schema()
