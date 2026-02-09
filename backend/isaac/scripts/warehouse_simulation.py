"""
Isaac Sim 창고 시뮬레이션 스크립트
Forge API에서 호출되어 실행됨
"""

import omni
from omni.isaac.kit import SimulationApp

# Headless 모드로 시작
CONFIG = {
    "headless": True,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
}

simulation_app = SimulationApp(CONFIG)

# Isaac Sim 모듈 import (SimulationApp 이후에 해야 함)
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np
import json
import asyncio


class WarehouseSimulation:
    """창고 시뮬레이션 관리자"""

    def __init__(self):
        self.world = None
        self.robots = []
        self.warehouse = None
        self.metrics = {
            "throughput": 0,
            "collisions": 0,
            "items_processed": 0
        }

    async def setup(self, config: dict):
        """
        시뮬레이션 환경 설정

        Args:
            config: {
                "warehouse": {"width": 30, "length": 20, "height": 5},
                "robots": [{"type": "AGV", "count": 3, "speed": 1.0}],
                "temperature": 25,
                "humidity": 50
            }
        """
        # World 생성
        self.world = World(stage_units_in_meters=1.0)
        await self.world.initialize_simulation_context_async()

        # 바닥 생성
        self.world.scene.add_default_ground_plane()

        # 창고 환경 생성
        warehouse_config = config.get("warehouse", {})
        await self._create_warehouse(warehouse_config)

        # 로봇 생성
        robots_config = config.get("robots", [])
        await self._create_robots(robots_config)

        # 물리 설정
        self.world.reset()

        return {"status": "ready", "robots": len(self.robots)}

    async def _create_warehouse(self, config: dict):
        """창고 환경 생성"""
        width = config.get("width", 30)
        length = config.get("length", 20)
        height = config.get("height", 5)

        assets_root = get_assets_root_path()

        # 벽 생성 (간단한 박스로)
        from omni.isaac.core.objects import FixedCuboid

        # 좌측 벽
        self.world.scene.add(FixedCuboid(
            prim_path="/World/Warehouse/WallLeft",
            name="wall_left",
            position=np.array([-width/2, 0, height/2]),
            scale=np.array([0.2, length, height]),
            color=np.array([0.5, 0.5, 0.5])
        ))

        # 우측 벽
        self.world.scene.add(FixedCuboid(
            prim_path="/World/Warehouse/WallRight",
            name="wall_right",
            position=np.array([width/2, 0, height/2]),
            scale=np.array([0.2, length, height]),
            color=np.array([0.5, 0.5, 0.5])
        ))

        # 선반 생성
        shelf_rows = config.get("shelf_rows", 2)
        shelf_spacing = length / (shelf_rows + 1)

        for i in range(shelf_rows):
            y_pos = -length/2 + shelf_spacing * (i + 1)
            self.world.scene.add(FixedCuboid(
                prim_path=f"/World/Warehouse/Shelf_{i}",
                name=f"shelf_{i}",
                position=np.array([0, y_pos, 1.5]),
                scale=np.array([width * 0.6, 1.0, 3.0]),
                color=np.array([0.6, 0.4, 0.2])
            ))

    async def _create_robots(self, robots_config: list):
        """로봇 생성"""
        assets_root = get_assets_root_path()

        robot_idx = 0
        for robot_spec in robots_config:
            robot_type = robot_spec.get("type", "AGV")
            count = robot_spec.get("count", 1)
            speed = robot_spec.get("speed", 1.0)

            for i in range(count):
                # 시작 위치 계산 (분산 배치)
                x_pos = -10 + (robot_idx % 5) * 5
                y_pos = -5 + (robot_idx // 5) * 3

                if robot_type.upper() == "AGV":
                    # Jetbot을 AGV 대용으로 사용
                    robot_usd = assets_root + "/Isaac/Robots/Jetbot/jetbot.usd"

                    robot_prim_path = f"/World/Robots/AGV_{robot_idx}"
                    add_reference_to_stage(robot_usd, robot_prim_path)

                    robot = self.world.scene.add(
                        WheeledRobot(
                            prim_path=robot_prim_path,
                            name=f"agv_{robot_idx}",
                            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                            create_robot=True,
                            position=np.array([x_pos, y_pos, 0.05])
                        )
                    )
                    robot._speed = speed
                    self.robots.append(robot)

                robot_idx += 1

    async def run(self, duration: float = 60.0, time_step: float = 1/60):
        """
        시뮬레이션 실행

        Args:
            duration: 시뮬레이션 시간 (초)
            time_step: 물리 스텝 간격

        Returns:
            시뮬레이션 결과
        """
        if not self.world:
            return {"error": "World not initialized"}

        total_steps = int(duration / time_step)
        self.metrics = {"throughput": 0, "collisions": 0, "items_processed": 0}

        for step in range(total_steps):
            # 로봇 제어 (간단한 랜덤 이동)
            for robot in self.robots:
                # 랜덤 속도 명령
                left_speed = np.random.uniform(0.5, 1.5) * robot._speed
                right_speed = np.random.uniform(0.5, 1.5) * robot._speed
                robot.apply_wheel_actions([left_speed, right_speed])

            # 물리 스텝
            self.world.step(render=False)

            # 메트릭 업데이트 (간략화)
            if step % 100 == 0:
                self.metrics["items_processed"] += len(self.robots)

            # 충돌 감지 (간략화)
            if step % 500 == 0 and np.random.random() < 0.1:
                self.metrics["collisions"] += 1

        # 처리량 계산
        self.metrics["throughput"] = self.metrics["items_processed"] / (duration / 3600)

        return {
            "success": True,
            "duration": duration,
            "metrics": self.metrics,
            "robot_count": len(self.robots)
        }

    def get_stream_url(self):
        """Livestream URL 반환"""
        return "http://localhost:8899/streaming/webrtc-client"

    def cleanup(self):
        """리소스 정리"""
        if self.world:
            self.world.stop()
            self.world.clear()


# 글로벌 인스턴스
_simulation = None


def get_simulation():
    global _simulation
    if _simulation is None:
        _simulation = WarehouseSimulation()
    return _simulation


# API 엔드포인트용 함수
async def setup_simulation(config: dict):
    sim = get_simulation()
    return await sim.setup(config)


async def run_simulation(duration: float = 60.0):
    sim = get_simulation()
    return await sim.run(duration)


def get_stream_url():
    sim = get_simulation()
    return sim.get_stream_url()


def cleanup_simulation():
    global _simulation
    if _simulation:
        _simulation.cleanup()
        _simulation = None


if __name__ == "__main__":
    # 테스트 실행
    import asyncio

    async def test():
        config = {
            "warehouse": {"width": 30, "length": 20, "height": 5},
            "robots": [{"type": "AGV", "count": 3, "speed": 1.0}]
        }

        result = await setup_simulation(config)
        print(f"Setup: {result}")

        result = await run_simulation(duration=10.0)
        print(f"Result: {result}")

        cleanup_simulation()

    asyncio.run(test())

    # 앱 종료
    simulation_app.close()
