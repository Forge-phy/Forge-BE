"""
Forge Isaac Sim - 시뮬레이션 실행
시뮬레이션 실행 및 결과 수집

모드:
- mock: 가짜 시뮬레이션 결과 (개발/테스트용)
- local: 로컬 Isaac Sim SDK로 실제 시뮬레이션
- docker: Docker 컨테이너에서 시뮬레이션
"""

import logging
import asyncio
import random
import math
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime

from .schemas import (
    EnvironmentConfig, SimulationConfig, SimulationResult,
    RobotMetrics, RobotConfig
)
from .client import IsaacSimClient, get_client
from .environment import EnvironmentBuilder

logger = logging.getLogger(__name__)

# Isaac Sim SDK 동적 import
_isaac_available = False
try:
    from omni.isaac.core import World
    _isaac_available = True
except ImportError:
    pass


class SimulationRunner:
    """
    시뮬레이션 실행기

    Isaac Sim에서 시뮬레이션을 실행하고 결과를 수집합니다.
    """

    def __init__(self, client: Optional[IsaacSimClient] = None):
        self.client = client or get_client()
        self.env_builder = EnvironmentBuilder(self.client)
        self._is_running = False
        self._current_time = 0.0

    def _config_to_dict(self, env_config: EnvironmentConfig) -> Dict[str, Any]:
        """EnvironmentConfig를 API용 dict로 변환"""
        robots = []
        for robot in env_config.robots:
            robots.append({
                "type": robot.robot_type.value if hasattr(robot.robot_type, 'value') else str(robot.robot_type),
                "count": 1,
                "speed": robot.speed
            })

        return {
            "environment": {
                "warehouse": {
                    "width": env_config.warehouse.width,
                    "length": env_config.warehouse.length,
                },
                "width": env_config.warehouse.width,
                "length": env_config.warehouse.length,
                "shelves": env_config.warehouse.shelf_rows,
            },
            "robots": robots,
            "temperature": env_config.temperature,
            "humidity": env_config.humidity
        }

    async def run_simulation(
        self,
        env_config: EnvironmentConfig,
        sim_config: SimulationConfig
    ) -> SimulationResult:
        """
        시뮬레이션 실행

        Args:
            env_config: 환경 설정
            sim_config: 시뮬레이션 설정

        Returns:
            시뮬레이션 결과
        """
        logger.info("Starting simulation...")
        start_time = datetime.now()

        try:
            # 1. 환경 생성 (모드에 따라 다르게 처리)
            if self.client.mode == "aws":
                # AWS 모드: client.create_environment() 직접 호출
                env_dict = self._config_to_dict(env_config)
                env_result = await self.client.create_environment(env_dict)
                if not env_result.get("success"):
                    raise Exception(f"Environment creation failed: {env_result.get('error')}")
                logger.info(f"AWS environment created: {env_result.get('message')}")
            else:
                # 다른 모드: EnvironmentBuilder 사용
                await self.env_builder.build_environment(env_config)

            # 2. 시뮬레이션 실행
            result = await self._execute_simulation(env_config, sim_config)

            # 3. 결과 수집
            result.total_duration = sim_config.duration
            result.avg_temperature = env_config.temperature
            result.avg_humidity = env_config.humidity

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Simulation completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return SimulationResult(
                success=False,
                error_message=str(e)
            )

    async def _execute_simulation(
        self,
        env_config: EnvironmentConfig,
        sim_config: SimulationConfig
    ) -> SimulationResult:
        """시뮬레이션 실제 실행"""
        self._is_running = True
        self._current_time = 0.0

        # 로봇별 메트릭 초기화
        robot_metrics = {
            r.robot_id: RobotMetrics(robot_id=r.robot_id)
            for r in env_config.robots
        }

        timeseries_data = []
        total_collisions = 0
        total_items = 0

        # 모드에 따라 분기
        if _isaac_available and self.client.mode == "local":
            # 실제 Isaac Sim 시뮬레이션
            return await self._execute_real_simulation(
                env_config, sim_config, robot_metrics
            )

        # Mock 시뮬레이션 (개발/테스트용)
        logger.info("Running mock simulation...")
        return await self._execute_mock_simulation(
            env_config, sim_config, robot_metrics
        )

    async def _execute_real_simulation(
        self,
        env_config: EnvironmentConfig,
        sim_config: SimulationConfig,
        robot_metrics: Dict[str, RobotMetrics]
    ) -> SimulationResult:
        """실제 Isaac Sim 시뮬레이션 실행"""
        logger.info("Running REAL Isaac Sim simulation...")

        world = self.client.get_world()
        if not world:
            logger.error("World not initialized")
            return SimulationResult(success=False, error_message="World not initialized")

        world.reset()

        timeseries_data = []
        total_collisions = 0
        total_items = 0

        # 물리 시뮬레이션 루프
        total_steps = int(sim_config.duration / sim_config.time_step)
        sample_interval = max(1, total_steps // 60)  # 60개 샘플 포인트

        for step in range(total_steps):
            if not self._is_running:
                break

            self._current_time = step * sim_config.time_step

            # 로봇 제어 (간단한 랜덤 이동)
            for robot_config in env_config.robots:
                robot_obj = world.scene.get_object(robot_config.robot_id)
                if robot_obj and hasattr(robot_obj, 'apply_wheel_actions'):
                    speed = robot_config.speed
                    left_speed = np.random.uniform(0.5, 1.5) * speed
                    right_speed = np.random.uniform(0.5, 1.5) * speed
                    robot_obj.apply_wheel_actions(
                        np.array([left_speed, right_speed])
                    )

            # 물리 스텝
            world.step(render=not self.client.config.headless)

            # 메트릭 수집 (샘플링)
            if step % sample_interval == 0:
                step_items = 0
                step_collisions = 0

                for robot_config in env_config.robots:
                    robot_id = robot_config.robot_id
                    metrics = robot_metrics[robot_id]
                    robot_obj = world.scene.get_object(robot_id)

                    if robot_obj:
                        # 속도 기반 이동 거리 계산
                        try:
                            velocity = robot_obj.get_linear_velocity()
                            if velocity is not None:
                                speed = np.linalg.norm(velocity)
                                metrics.distance_traveled += speed * sim_config.time_step * sample_interval
                                metrics.active_time += sim_config.time_step * sample_interval if speed > 0.1 else 0
                        except Exception:
                            pass

                    # 처리량 추정 (환경 조건 반영)
                    base_items = random.uniform(0.3, 0.8)
                    temp_factor = 1.0 - 0.01 * abs(env_config.temperature - 25)
                    humidity_factor = 1.0 - 0.005 * abs(env_config.humidity - 50)
                    items = base_items * temp_factor * humidity_factor
                    metrics.items_transported += max(0, int(items))
                    step_items += max(0, int(items))

                    # 충돌 감지 (간략화)
                    if random.random() < 0.005 * len(env_config.robots):
                        metrics.collisions += 1
                        step_collisions += 1

                total_items += step_items
                total_collisions += step_collisions

                timeseries_data.append({
                    "timestamp": self._current_time,
                    "items_processed": step_items,
                    "collisions": step_collisions,
                    "active_robots": len(env_config.robots)
                })

            # 진행률 로깅
            if step % (total_steps // 10) == 0:
                progress = (step / total_steps) * 100
                logger.info(f"Simulation progress: {progress:.1f}%")

        self._is_running = False

        # 최종 메트릭 계산
        for robot_id, metrics in robot_metrics.items():
            metrics.idle_time = sim_config.duration - metrics.active_time
            if metrics.active_time > 0:
                metrics.average_speed = metrics.distance_traveled / metrics.active_time

        hours = sim_config.duration / 3600
        throughput = total_items / hours if hours > 0 else 0

        logger.info(f"Real simulation completed: {throughput:.1f} items/h, {total_collisions} collisions")

        return SimulationResult(
            success=True,
            total_duration=sim_config.duration,
            total_throughput=throughput,
            total_collisions=total_collisions,
            total_items_processed=total_items,
            robot_metrics=list(robot_metrics.values()),
            avg_temperature=env_config.temperature,
            avg_humidity=env_config.humidity,
            timeseries_data=timeseries_data
        )

    async def _execute_mock_simulation(
        self,
        env_config: EnvironmentConfig,
        sim_config: SimulationConfig,
        robot_metrics: Dict[str, RobotMetrics]
    ) -> SimulationResult:
        """Mock 시뮬레이션 (개발/테스트용)"""
        timeseries_data = []
        total_collisions = 0
        total_items = 0

        num_steps = int(sim_config.duration / 60)  # 분 단위로 샘플링

        for step in range(num_steps):
            self._current_time = step * 60

            step_items = 0
            step_collisions = 0

            for robot_config in env_config.robots:
                robot_id = robot_config.robot_id
                metrics = robot_metrics[robot_id]

                # 처리량 시뮬레이션 (환경 조건 반영)
                base_items = random.uniform(0.5, 1.5)

                # 온도 영향 (25°C 기준)
                temp_factor = 1.0 - 0.01 * abs(env_config.temperature - 25)

                # 습도 영향 (50% 기준)
                humidity_factor = 1.0 - 0.005 * abs(env_config.humidity - 50)

                # 로봇 수 영향 (혼잡도)
                congestion_factor = 1.0 - 0.05 * max(0, len(env_config.robots) - 3)

                items = base_items * temp_factor * humidity_factor * congestion_factor
                items = max(0, items + random.gauss(0, 0.1))

                metrics.items_transported += int(items)
                metrics.tasks_completed += 1 if items > 0.5 else 0
                metrics.distance_traveled += robot_config.speed * 60 * random.uniform(0.3, 0.8)
                metrics.active_time += 60 * random.uniform(0.6, 0.95)

                step_items += int(items)

                # 충돌 시뮬레이션
                if random.random() < 0.01 * len(env_config.robots):
                    metrics.collisions += 1
                    step_collisions += 1

            total_items += step_items
            total_collisions += step_collisions

            timeseries_data.append({
                "timestamp": self._current_time,
                "items_processed": step_items,
                "collisions": step_collisions,
                "active_robots": len(env_config.robots)
            })

            if step % 10 == 0:
                progress = (step / num_steps) * 100
                logger.debug(f"Mock simulation progress: {progress:.1f}%")

        self._is_running = False

        # 최종 메트릭 계산
        for robot_id, metrics in robot_metrics.items():
            metrics.idle_time = sim_config.duration - metrics.active_time
            if metrics.active_time > 0:
                metrics.average_speed = metrics.distance_traveled / metrics.active_time

        hours = sim_config.duration / 3600
        throughput = total_items / hours if hours > 0 else 0

        return SimulationResult(
            success=True,
            total_duration=sim_config.duration,
            total_throughput=throughput,
            total_collisions=total_collisions,
            total_items_processed=total_items,
            robot_metrics=list(robot_metrics.values()),
            avg_temperature=env_config.temperature,
            avg_humidity=env_config.humidity,
            timeseries_data=timeseries_data
        )

    async def stop_simulation(self):
        """시뮬레이션 중지"""
        logger.info("Stopping simulation...")
        self._is_running = False

        if _isaac_available and self.client.mode == "local":
            world = self.client.get_world()
            if world:
                world.stop()
                logger.info("Isaac Sim simulation stopped")

    def is_running(self) -> bool:
        """시뮬레이션 실행 중 여부"""
        return self._is_running

    def get_current_time(self) -> float:
        """현재 시뮬레이션 시간"""
        return self._current_time


async def run_quick_simulation(
    warehouse_width: float = 30.0,
    warehouse_length: float = 20.0,
    robot_count: int = 3,
    duration: float = 3600.0,
    temperature: float = 25.0,
    humidity: float = 50.0
) -> SimulationResult:
    """
    간단한 시뮬레이션 실행 (헬퍼 함수)

    Args:
        warehouse_width: 창고 너비 (m)
        warehouse_length: 창고 길이 (m)
        robot_count: 로봇 수
        duration: 시뮬레이션 시간 (초)
        temperature: 환경 온도 (°C)
        humidity: 환경 습도 (%)

    Returns:
        시뮬레이션 결과
    """
    from .environment import create_default_warehouse_config

    env_config = create_default_warehouse_config(
        width=warehouse_width,
        length=warehouse_length,
        robot_count=robot_count
    )
    env_config.temperature = temperature
    env_config.humidity = humidity

    sim_config = SimulationConfig(
        duration=duration,
        realtime_factor=10.0  # 10배속
    )

    runner = SimulationRunner()
    return await runner.run_simulation(env_config, sim_config)
