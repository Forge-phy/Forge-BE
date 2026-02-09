"""
Forge Isaac Sim - 환경 생성
창고, 로봇 등 시뮬레이션 환경 생성

모드:
- mock: 가짜 데이터 반환 (개발/테스트용)
- local: 로컬 Isaac Sim SDK 직접 사용
- docker: Docker 컨테이너 내 Isaac Sim 사용
"""

import logging
from typing import Optional, List, Dict, Any
import math
import numpy as np

from .schemas import (
    EnvironmentConfig, WarehouseConfig, RobotConfig,
    RobotType, Position, Rotation
)
from .client import IsaacSimClient, get_client

logger = logging.getLogger(__name__)

# Isaac Sim SDK 동적 import (로컬 모드에서만 사용)
_isaac_available = False
try:
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.wheeled_robots.robots import WheeledRobot
    from pxr import UsdGeom, Gf
    _isaac_available = True
    logger.info("Isaac Sim SDK available")
except ImportError:
    logger.warning("Isaac Sim SDK not available - running in mock/docker mode only")


class EnvironmentBuilder:
    """
    시뮬레이션 환경 빌더

    자연어 또는 설정 객체로부터 Isaac Sim 환경을 생성합니다.
    """

    def __init__(self, client: Optional[IsaacSimClient] = None):
        self.client = client or get_client()
        self._prims = []  # 생성된 USD prim 목록
        self._robots = []  # 생성된 로봇 목록

    async def build_environment(self, config: EnvironmentConfig) -> bool:
        """
        환경 설정으로부터 시뮬레이션 환경 생성

        Args:
            config: 환경 설정

        Returns:
            생성 성공 여부
        """
        if not self.client.is_connected():
            logger.error("Isaac Sim not connected")
            return False

        try:
            # 1. 기본 환경 생성 (바닥, 조명)
            await self._create_base_environment()

            # 2. 창고 구조물 생성
            await self._create_warehouse(config.warehouse)

            # 3. 로봇 배치
            for robot_config in config.robots:
                await self._spawn_robot(robot_config)

            logger.info(f"Environment created: {len(self._robots)} robots")
            return True

        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            return False

    async def _create_base_environment(self):
        """기본 환경 생성 (바닥, 조명, 물리)"""
        logger.info("Creating base environment...")

        if self.client.is_mock:
            # Mock 모드 - 가짜 prim 추가
            self._prims.append("/World/GroundPlane")
            self._prims.append("/World/DistantLight")
            return

        # 실제 Isaac Sim SDK 사용 (local 모드)
        if _isaac_available and self.client.mode == "local":
            world = self.client.get_world()
            if world:
                # Ground plane
                world.scene.add_default_ground_plane()
                self._prims.append("/World/defaultGroundPlane")

                # Distant light
                create_prim(
                    prim_path="/World/DistantLight",
                    prim_type="DistantLight",
                    attributes={"intensity": 3000.0}
                )
                self._prims.append("/World/DistantLight")
                logger.info("Base environment created with Isaac Sim SDK")
        else:
            # Docker 모드는 컨테이너에서 처리
            self._prims.append("/World/GroundPlane")
            self._prims.append("/World/DistantLight")

    async def _create_warehouse(self, config: WarehouseConfig):
        """창고 구조물 생성"""
        logger.info(f"Creating warehouse: {config.width}m x {config.length}m")

        # 벽 생성
        await self._create_walls(config)

        # 선반 생성
        await self._create_shelves(config)

    async def _create_walls(self, config: WarehouseConfig):
        """창고 벽 생성"""
        wall_thickness = 0.2
        wall_positions = [
            # (x, y, z, width, length, height) - 4면
            (config.width/2, 0, config.height/2, wall_thickness, config.length, config.height),  # +X wall
            (-config.width/2, 0, config.height/2, wall_thickness, config.length, config.height), # -X wall
            (0, config.length/2, config.height/2, config.width, wall_thickness, config.height),  # +Y wall
            (0, -config.length/2, config.height/2, config.width, wall_thickness, config.height), # -Y wall
        ]

        if _isaac_available and self.client.mode == "local":
            world = self.client.get_world()
            if world:
                for i, (x, y, z, w, l, h) in enumerate(wall_positions):
                    prim_path = f"/World/Warehouse/Wall_{i}"
                    world.scene.add(FixedCuboid(
                        prim_path=prim_path,
                        name=f"wall_{i}",
                        position=np.array([x, y, z]),
                        scale=np.array([w, l, h]),
                        color=np.array([0.5, 0.5, 0.5])
                    ))
                    self._prims.append(prim_path)
        else:
            # Mock/Docker 모드
            for i, (x, y, z, w, l, h) in enumerate(wall_positions):
                prim_path = f"/World/Warehouse/Wall_{i}"
                self._prims.append(prim_path)

    async def _create_shelves(self, config: WarehouseConfig):
        """선반 생성"""
        shelf_width = 1.0
        shelf_depth = 0.5
        shelf_height = 2.0

        x_spacing = (config.width - 2 * config.aisle_width) / max(1, config.shelf_columns)
        y_spacing = (config.length - 2 * config.aisle_width) / max(1, config.shelf_rows)

        if _isaac_available and self.client.mode == "local":
            world = self.client.get_world()
            if world:
                for row in range(config.shelf_rows):
                    for col in range(config.shelf_columns):
                        x = -config.width/2 + config.aisle_width + col * x_spacing + x_spacing/2
                        y = -config.length/2 + config.aisle_width + row * y_spacing + y_spacing/2

                        prim_path = f"/World/Warehouse/Shelf_{row}_{col}"
                        world.scene.add(FixedCuboid(
                            prim_path=prim_path,
                            name=f"shelf_{row}_{col}",
                            position=np.array([x, y, shelf_height/2]),
                            scale=np.array([shelf_width, shelf_depth, shelf_height]),
                            color=np.array([0.6, 0.4, 0.2])
                        ))
                        self._prims.append(prim_path)
        else:
            # Mock/Docker 모드
            for row in range(config.shelf_rows):
                for col in range(config.shelf_columns):
                    prim_path = f"/World/Warehouse/Shelf_{row}_{col}"
                    self._prims.append(prim_path)

        logger.info(f"Created {config.shelf_rows * config.shelf_columns} shelves")

    async def _spawn_robot(self, config: RobotConfig):
        """로봇 스폰"""
        logger.info(f"Spawning robot: {config.robot_id} ({config.robot_type.value})")

        # 로봇 타입별 USD 경로 (Isaac Sim 기본 에셋)
        robot_usd_paths = {
            RobotType.AGV: "/Isaac/Robots/Jetbot/jetbot.usd",  # AGV 대용
            RobotType.AMR: "/Isaac/Robots/Jetbot/jetbot.usd",
            RobotType.FORKLIFT: "/Isaac/Robots/Forklift/forklift.usd",
            RobotType.ARM: "/Isaac/Robots/Franka/franka.usd",
        }

        prim_path = f"/World/Robots/{config.robot_id}"

        if _isaac_available and self.client.mode == "local":
            world = self.client.get_world()
            if world:
                try:
                    assets_root = get_assets_root_path()
                    usd_path = assets_root + robot_usd_paths.get(
                        config.robot_type, robot_usd_paths[RobotType.AGV]
                    )

                    # USD 파일 로드
                    add_reference_to_stage(usd_path, prim_path)

                    # WheeledRobot으로 로봇 추가 (AGV/AMR용)
                    if config.robot_type in [RobotType.AGV, RobotType.AMR]:
                        robot = world.scene.add(
                            WheeledRobot(
                                prim_path=prim_path,
                                name=config.robot_id,
                                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                                create_robot=True,
                                position=np.array([
                                    config.position.x,
                                    config.position.y,
                                    config.position.z + 0.05  # 바닥 위에 배치
                                ])
                            )
                        )
                        robot._speed = config.speed
                        logger.info(f"Robot {config.robot_id} spawned with Isaac Sim SDK")
                    else:
                        # 위치 설정 (다른 로봇 타입)
                        stage = self.client.get_stage()
                        if stage:
                            prim = stage.GetPrimAtPath(prim_path)
                            if prim:
                                xform = UsdGeom.Xformable(prim)
                                xform.AddTranslateOp().Set(Gf.Vec3d(
                                    config.position.x,
                                    config.position.y,
                                    config.position.z
                                ))

                except Exception as e:
                    logger.error(f"Failed to spawn robot with SDK: {e}")

        # 로봇 정보 저장 (모든 모드)
        self._robots.append({
            "robot_id": config.robot_id,
            "prim_path": prim_path,
            "config": config
        })

    def get_robots(self) -> List[Dict[str, Any]]:
        """생성된 로봇 목록 반환"""
        return self._robots

    def get_prims(self) -> List[str]:
        """생성된 prim 목록 반환"""
        return self._prims

    async def clear_environment(self):
        """환경 초기화"""
        logger.info("Clearing environment...")

        if _isaac_available and self.client.mode == "local":
            world = self.client.get_world()
            if world:
                world.clear()
                logger.info("Isaac Sim world cleared")

        self._prims = []
        self._robots = []


def create_default_warehouse_config(
    width: float = 30.0,
    length: float = 20.0,
    robot_count: int = 3
) -> EnvironmentConfig:
    """
    기본 창고 환경 설정 생성

    Args:
        width: 창고 너비 (m)
        length: 창고 길이 (m)
        robot_count: 로봇 수

    Returns:
        환경 설정
    """
    warehouse = WarehouseConfig(
        width=width,
        length=length,
        height=5.0,
        shelf_rows=2,
        shelf_columns=5,
        aisle_width=3.0
    )

    # 로봇 배치 (입구 근처에 일렬로)
    robots = []
    for i in range(robot_count):
        robots.append(RobotConfig(
            robot_id=f"AGV_{i+1}",
            robot_type=RobotType.AGV,
            position=Position(
                x=-warehouse.width/2 + 2.0,
                y=-warehouse.length/2 + 2.0 + i * 2.0,
                z=0.0
            ),
            speed=1.5,
            payload_capacity=100.0
        ))

    return EnvironmentConfig(
        warehouse=warehouse,
        robots=robots,
        temperature=25.0,
        humidity=50.0
    )
