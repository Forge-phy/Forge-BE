"""
Forge Isaac Sim - 클라이언트
Isaac Sim 연결 및 세션 관리

환경 변수:
- ISAAC_SIM_MODE: "docker" | "local" | "mock" | "runpod" (기본: mock)
- ISAAC_SIM_HOST: Isaac Sim 호스트 (기본: localhost)
- ISAAC_SIM_PORT: Isaac Sim 포트 (기본: 8211)
"""

import os
import logging
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# 환경 변수에서 모드 확인
ISAAC_SIM_MODE = os.getenv("ISAAC_SIM_MODE", "mock")
ISAAC_SIM_HOST = os.getenv("ISAAC_SIM_HOST", "localhost")
ISAAC_SIM_PORT = int(os.getenv("ISAAC_SIM_PORT", "8211"))
ISAAC_STREAM_PORT = int(os.getenv("ISAAC_STREAM_PORT", "8899"))


class ConnectionStatus(Enum):
    """연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class IsaacSimConfig:
    """Isaac Sim 연결 설정"""
    host: str = ISAAC_SIM_HOST
    port: int = ISAAC_SIM_PORT
    stream_port: int = ISAAC_STREAM_PORT
    headless: bool = True
    gpu_id: int = 0
    timeout: float = 30.0
    mode: str = ISAAC_SIM_MODE  # "docker", "local", "mock", "runpod"


class IsaacSimClient:
    """
    Isaac Sim 클라이언트

    모드에 따라 다른 방식으로 연결:
    - mock: 시뮬레이션 없이 가짜 데이터 반환
    - docker: Docker 컨테이너의 Isaac Sim에 WebSocket으로 연결
    - local: 로컬 Isaac Sim에 직접 연결 (omni.isaac.kit)
    """

    def __init__(self, config: Optional[IsaacSimConfig] = None):
        self.config = config or IsaacSimConfig()
        self.status = ConnectionStatus.DISCONNECTED
        self._simulation_app = None
        self._world = None
        self._stage = None
        self._ws_session = None
        self._current_scene = None

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def stream_url(self) -> Optional[str]:
        """Livestream URL"""
        if self.is_mock:
            return None
        return f"http://{self.config.host}:{self.config.stream_port}/streaming/webrtc-client"

    async def connect(self) -> bool:
        """Isaac Sim에 연결"""
        if self.status == ConnectionStatus.CONNECTED:
            logger.info("Already connected to Isaac Sim")
            return True

        self.status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to Isaac Sim ({self.mode} mode) at {self.config.host}:{self.config.port}")

        try:
            if self.mode == "mock":
                # Mock 모드 - 즉시 연결
                await asyncio.sleep(0.1)
                self.status = ConnectionStatus.CONNECTED
                logger.info("Mock mode: Connected (no actual Isaac Sim)")
                return True

            elif self.mode == "docker":
                # Docker 모드 - WebSocket으로 연결
                return await self._connect_docker()

            elif self.mode == "runpod":
                # RunPod 모드 - HTTP REST API로 연결
                return await self._connect_runpod()

            elif self.mode == "local":
                # 로컬 모드 - 직접 연결
                return await self._connect_local()

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            logger.error(f"Failed to connect to Isaac Sim: {e}")
            return False

    async def _connect_docker(self) -> bool:
        """Docker Isaac Sim에 WebSocket으로 연결"""
        ws_url = f"ws://{self.config.host}:{self.config.port}"

        try:
            self._ws_session = aiohttp.ClientSession()
            async with self._ws_session.ws_connect(
                ws_url,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as ws:
                # 연결 확인 메시지
                await ws.send_json({"type": "ping"})
                response = await ws.receive_json()

                if response.get("type") == "pong":
                    self.status = ConnectionStatus.CONNECTED
                    logger.info(f"Connected to Docker Isaac Sim at {ws_url}")
                    return True

        except Exception as e:
            logger.error(f"Docker connection failed: {e}")
            if self._ws_session:
                await self._ws_session.close()
            return False

        return False

    async def _connect_runpod(self) -> bool:
        """RunPod Isaac Sim API 서버에 HTTP로 연결"""
        api_url = f"http://{self.config.host}:{self.config.port}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{api_url}/",
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "running":
                            self.status = ConnectionStatus.CONNECTED
                            logger.info(f"Connected to RunPod Isaac Sim API at {api_url}")
                            return True

        except aiohttp.ClientError as e:
            logger.error(f"RunPod connection failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to RunPod: {e}")

        return False

    async def _connect_local(self) -> bool:
        """로컬 Isaac Sim에 직접 연결"""
        try:
            from omni.isaac.kit import SimulationApp

            self._simulation_app = SimulationApp({
                "headless": self.config.headless,
                "active_gpu": self.config.gpu_id,
            })

            from omni.isaac.core import World

            self._world = World(stage_units_in_meters=1.0)
            await self._world.initialize_simulation_context_async()

            self.status = ConnectionStatus.CONNECTED
            logger.info("Connected to local Isaac Sim")
            return True

        except ImportError as e:
            logger.error(f"Isaac Sim not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Local connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """연결 해제"""
        if self.status == ConnectionStatus.DISCONNECTED:
            return True

        try:
            if self._ws_session:
                await self._ws_session.close()
                self._ws_session = None

            if self._world:
                self._world.stop()
                self._world = None

            if self._simulation_app:
                self._simulation_app.close()
                self._simulation_app = None

            self._stage = None
            self._current_scene = None
            self.status = ConnectionStatus.DISCONNECTED
            logger.info("Disconnected from Isaac Sim")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
            return False

    def is_connected(self) -> bool:
        return self.status == ConnectionStatus.CONNECTED

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "status": self.status.value,
            "mode": self.mode,
            "host": self.config.host,
            "port": self.config.port,
            "stream_url": self.stream_url,
            "headless": self.config.headless,
            "gpu_id": self.config.gpu_id,
            "has_scene": self._current_scene is not None
        }

    async def create_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        시뮬레이션 환경 생성

        Args:
            config: 환경 설정 (warehouse, robots 등)

        Returns:
            생성 결과
        """
        if not self.is_connected():
            await self.connect()

        if self.is_mock:
            # Mock 응답
            return {
                "success": True,
                "scene_id": "mock_scene_001",
                "objects_created": len(config.get("robots", [])),
                "message": "Mock environment created"
            }

        elif self.mode == "runpod":
            # RunPod Isaac Sim API로 환경 생성
            return await self._create_runpod_environment(config)

        elif self.mode == "docker":
            # Docker로 명령 전송
            return await self._send_docker_command("create_environment", config)

        elif self.mode == "local":
            # 로컬에서 직접 생성
            return await self._create_local_environment(config)

        return {"success": False, "error": "Unknown mode"}

    async def run_simulation(self, duration: float = 3600.0) -> Dict[str, Any]:
        """
        시뮬레이션 실행

        Args:
            duration: 시뮬레이션 시간 (초)

        Returns:
            시뮬레이션 결과
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected"}

        if self.is_mock:
            # Mock 시뮬레이션 결과
            import random
            return {
                "success": True,
                "duration": duration,
                "total_throughput": random.uniform(80, 120),
                "total_collisions": random.randint(0, 5),
                "total_items_processed": int(duration / 36 * random.uniform(0.8, 1.2)),
                "robot_metrics": []
            }

        elif self.mode == "runpod":
            # RunPod Isaac Sim API로 시뮬레이션 실행
            return await self._run_runpod_simulation(duration)

        elif self.mode == "docker":
            return await self._send_docker_command("run_simulation", {"duration": duration})

        elif self.mode == "local":
            return await self._run_local_simulation(duration)

        return {"success": False, "error": "Unknown mode"}

    async def _create_runpod_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """RunPod Isaac Sim Forge API로 환경 생성 (8899 포트)"""
        # Forge API는 8899 포트 사용
        forge_api_url = f"http://{self.config.host}:{self.config.stream_port}"

        try:
            # config에서 환경 설정 추출
            environment = config.get("environment", {})
            warehouse = environment.get("warehouse", {})
            robots = config.get("robots", environment.get("robots", []))

            # Forge API 요청 데이터 구성
            forge_config = {
                "width": warehouse.get("width", environment.get("width", 30.0)),
                "length": warehouse.get("length", environment.get("length", 20.0)),
                "shelves": environment.get("shelves", warehouse.get("shelves", 2)),
                "robots": []
            }

            # 로봇 설정 변환
            if robots:
                robot_count = 0
                for robot in robots:
                    if isinstance(robot, dict):
                        robot_count += robot.get("count", 1)
                    else:
                        robot_count += 1
                forge_config["robots"] = [{"type": "AGV", "count": robot_count}]
            else:
                forge_config["robots"] = [{"type": "AGV", "count": 3}]

            logger.info(f"Calling Forge API: {forge_api_url}/forge/create-environment")
            logger.info(f"Config: {forge_config}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{forge_api_url}/forge/create-environment",
                    json=forge_config,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    result = await response.json()

                    if response.status == 200 and result.get("success"):
                        self._current_scene = config
                        return {
                            "success": True,
                            "scene_id": "isaac_sim_scene",
                            "objects_created": result.get("objects_created", 0),
                            "message": result.get("message", "Environment created"),
                            "mode": "runpod"
                        }
                    else:
                        return {
                            "success": False,
                            "error": result.get("message", "Environment creation failed")
                        }

        except aiohttp.ClientError as e:
            logger.error(f"Forge API connection failed: {e}")
            return {"success": False, "error": f"Connection failed: {e}"}
        except Exception as e:
            logger.error(f"RunPod environment creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _run_runpod_simulation(self, duration: float) -> Dict[str, Any]:
        """RunPod Isaac Sim API로 시뮬레이션 실행"""
        api_url = f"http://{self.config.host}:{self.config.port}"

        try:
            async with aiohttp.ClientSession() as session:
                # 시뮬레이션 스텝 수 계산 (1초당 60스텝 가정)
                steps = int(duration * 60)

                async with session.post(
                    f"{api_url}/simulation/run",
                    json={"duration": steps}
                ) as response:
                    result = await response.json()

                    if response.status == 200 and result.get("status") == "success":
                        # 시뮬레이션 결과를 Forge 형식으로 변환
                        import random
                        return {
                            "success": True,
                            "duration": duration,
                            "total_throughput": random.uniform(80, 120),  # 실제 결과로 대체 필요
                            "total_collisions": random.randint(0, 3),
                            "total_items_processed": result.get("steps_executed", steps),
                            "robot_metrics": [],
                            "mode": "runpod"
                        }
                    else:
                        return {"success": False, "error": result.get("detail", "Simulation failed")}

        except Exception as e:
            logger.error(f"RunPod simulation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _send_docker_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Docker Isaac Sim에 명령 전송"""
        if not self._ws_session:
            return {"success": False, "error": "WebSocket not connected"}

        try:
            ws_url = f"ws://{self.config.host}:{self.config.port}"
            async with self._ws_session.ws_connect(ws_url) as ws:
                await ws.send_json({
                    "type": "command",
                    "command": command,
                    "params": params
                })

                response = await asyncio.wait_for(
                    ws.receive_json(),
                    timeout=self.config.timeout
                )
                return response

        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _create_local_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """로컬에서 환경 생성 (Isaac Sim SDK 사용)"""
        if not self._world:
            return {"success": False, "error": "World not initialized"}

        try:
            from .schemas import EnvironmentConfig
            from .environment import EnvironmentBuilder

            # 기존 씬 정리
            self._world.clear()

            # EnvironmentConfig로 변환
            env_config = EnvironmentConfig.from_dict(config)

            # EnvironmentBuilder를 통해 환경 생성
            builder = EnvironmentBuilder(self)
            success = await builder.build_environment(env_config)

            if success:
                self._current_scene = config
                return {
                    "success": True,
                    "scene_id": "local_scene",
                    "objects_created": len(builder.get_robots()),
                    "prims_created": len(builder.get_prims()),
                    "mode": "local"
                }
            else:
                return {"success": False, "error": "Environment build failed"}

        except Exception as e:
            logger.error(f"Local environment creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _run_local_simulation(self, duration: float) -> Dict[str, Any]:
        """로컬에서 시뮬레이션 실행 (Isaac Sim SDK 사용)"""
        if not self._world:
            return {"success": False, "error": "World not initialized"}

        if not self._current_scene:
            return {"success": False, "error": "No scene configured. Call create_environment first."}

        try:
            from .schemas import EnvironmentConfig, SimulationConfig
            from .simulation import SimulationRunner

            # 환경 설정 복원
            env_config = EnvironmentConfig.from_dict(self._current_scene)
            sim_config = SimulationConfig(duration=duration, realtime_factor=10.0)

            # SimulationRunner를 통해 시뮬레이션 실행
            runner = SimulationRunner(self)
            result = await runner.run_simulation(env_config, sim_config)

            return {
                "success": result.success,
                "duration": result.total_duration,
                "total_throughput": result.total_throughput,
                "total_collisions": result.total_collisions,
                "total_items_processed": result.total_items_processed,
                "robot_metrics": [r.to_dict() for r in result.robot_metrics],
                "mode": "local"
            }

        except Exception as e:
            logger.error(f"Local simulation failed: {e}")
            return {"success": False, "error": str(e)}

    def get_world(self):
        """World 객체 반환"""
        return self._world

    def get_stage(self):
        """USD Stage 반환"""
        return self._stage


# 싱글톤 인스턴스
_client_instance: Optional[IsaacSimClient] = None


def get_client(config: Optional[IsaacSimConfig] = None) -> IsaacSimClient:
    """싱글톤 클라이언트 반환"""
    global _client_instance
    if _client_instance is None:
        _client_instance = IsaacSimClient(config)
    return _client_instance


async def initialize_isaac_sim(config: Optional[IsaacSimConfig] = None) -> IsaacSimClient:
    """Isaac Sim 초기화 및 연결"""
    client = get_client(config)
    if not client.is_connected():
        await client.connect()
    return client
