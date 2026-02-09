#!/usr/bin/env python3
"""
Isaac Sim WebSocket Server
Docker 컨테이너 내부에서 실행되어 외부 API 명령을 처리

실행: python isaac_server.py
포트: 8211 (WebSocket), 8899 (Livestream)
"""

import asyncio
import json
import logging
import signal
import sys
from typing import Dict, Any, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("isaac_server")

# ============================================================
# Isaac Sim 초기화 (반드시 다른 import 전에)
# ============================================================

logger.info("Initializing Isaac Sim...")

from omni.isaac.kit import SimulationApp

CONFIG = {
    "headless": False,  # Livestream을 위해 headless=False
    "width": 1920,
    "height": 1080,
    "renderer": "RayTracedLighting",
    "enable_livestream": True,
    "livestream": {
        "port": 8899,
        "mode": "webrtc"
    }
}

simulation_app = SimulationApp(CONFIG)
logger.info("Isaac Sim initialized")

# Isaac Sim 모듈 import (SimulationApp 이후에)
import omni
from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
import numpy as np

# Livestream 활성화
enable_extension("omni.kit.livestream.webrtc")

# WebSocket
import websockets

# ============================================================
# 시뮬레이션 관리 클래스
# ============================================================

class IsaacSimManager:
    """Isaac Sim 시뮬레이션 관리자"""

    def __init__(self):
        self.world: Optional[World] = None
        self.robots = []
        self.obstacles = []
        self.current_config = None
        self.is_running = False
        self.metrics = {
            "throughput": 0,
            "collisions": 0,
            "items_processed": 0
        }

    async def initialize(self):
        """World 초기화"""
        logger.info("Initializing World...")
        self.world = World(stage_units_in_meters=1.0)
        await self.world.initialize_simulation_context_async()
        self.world.scene.add_default_ground_plane()
        logger.info("World initialized")
        return {"status": "ready"}

    async def create_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        환경 생성

        Args:
            config: {
                "environment": {"type": "warehouse", "width": 30, "length": 20},
                "robots": [{"type": "AGV", "count": 3}],
                "obstacles": [{"type": "shelf", "count": 2}]
            }
        """
        if not self.world:
            await self.initialize()

        logger.info(f"Creating environment: {json.dumps(config, indent=2)}")

        try:
            # 기존 씬 정리
            self._clear_scene()

            self.current_config = config
            env = config.get("environment", {})

            # 창고 벽 생성
            await self._create_walls(env)

            # 선반/장애물 생성
            await self._create_obstacles(config.get("obstacles", []), env)

            # 로봇 생성
            await self._create_robots(config.get("robots", []), env)

            # 물리 초기화
            self.world.reset()

            result = {
                "success": True,
                "scene_id": "scene_001",
                "objects_created": len(self.robots) + len(self.obstacles),
                "robots": len(self.robots),
                "obstacles": len(self.obstacles),
                "message": "Environment created successfully"
            }
            logger.info(f"Environment created: {result}")
            return result

        except Exception as e:
            logger.error(f"Error creating environment: {e}")
            return {"success": False, "error": str(e)}

    def _clear_scene(self):
        """씬 정리"""
        self.robots.clear()
        self.obstacles.clear()

        if self.world:
            # 기존 객체 제거 (바닥 제외)
            stage = omni.usd.get_context().get_stage()
            for prim in stage.GetPseudoRoot().GetChildren():
                path = str(prim.GetPath())
                if path.startswith("/World/") and "Ground" not in path:
                    try:
                        stage.RemovePrim(path)
                    except:
                        pass

    async def _create_walls(self, env: Dict[str, Any]):
        """창고 벽 생성"""
        width = env.get("width", 30)
        length = env.get("length", 20)
        height = env.get("height", 5)

        wall_color = np.array([0.4, 0.4, 0.45])
        wall_thickness = 0.3

        # 4면 벽 생성
        walls = [
            # 좌측 벽
            ("/World/Walls/WallLeft", np.array([-width/2, 0, height/2]),
             np.array([wall_thickness, length + wall_thickness*2, height])),
            # 우측 벽
            ("/World/Walls/WallRight", np.array([width/2, 0, height/2]),
             np.array([wall_thickness, length + wall_thickness*2, height])),
            # 전면 벽
            ("/World/Walls/WallFront", np.array([0, length/2, height/2]),
             np.array([width, wall_thickness, height])),
            # 후면 벽 (입구 있음 - 절반만)
            ("/World/Walls/WallBackL", np.array([-width/4, -length/2, height/2]),
             np.array([width/2 - 3, wall_thickness, height])),
            ("/World/Walls/WallBackR", np.array([width/4, -length/2, height/2]),
             np.array([width/2 - 3, wall_thickness, height])),
        ]

        for prim_path, position, scale in walls:
            self.world.scene.add(FixedCuboid(
                prim_path=prim_path,
                name=prim_path.split("/")[-1],
                position=position,
                scale=scale,
                color=wall_color
            ))

        logger.info(f"Created walls for {width}m x {length}m warehouse")

    async def _create_obstacles(self, obstacles_config: list, env: Dict[str, Any]):
        """선반/장애물 생성"""
        width = env.get("width", 30)
        length = env.get("length", 20)

        shelf_color = np.array([0.6, 0.4, 0.2])

        for obs_spec in obstacles_config:
            obs_type = obs_spec.get("type", "shelf")
            count = obs_spec.get("count", 2)
            spacing = obs_spec.get("spacing", length / (count + 1))

            for i in range(count):
                y_pos = -length/2 + spacing * (i + 1)

                prim_path = f"/World/Obstacles/{obs_type}_{i}"
                self.world.scene.add(FixedCuboid(
                    prim_path=prim_path,
                    name=f"{obs_type}_{i}",
                    position=np.array([0, y_pos, 1.5]),
                    scale=np.array([width * 0.6, 1.2, 3.0]),
                    color=shelf_color
                ))
                self.obstacles.append(prim_path)

        logger.info(f"Created {len(self.obstacles)} obstacles")

    async def _create_robots(self, robots_config: list, env: Dict[str, Any]):
        """로봇 생성"""
        width = env.get("width", 30)
        length = env.get("length", 20)

        assets_root = get_assets_root_path()
        robot_idx = 0

        for robot_spec in robots_config:
            robot_type = robot_spec.get("type", "AGV")
            count = robot_spec.get("count", 1)
            speed = robot_spec.get("speed", 1.5)

            for i in range(count):
                # 입구 근처에 분산 배치
                x_pos = -width/4 + (robot_idx % 3) * (width/6)
                y_pos = -length/2 + 3 + (robot_idx // 3) * 2

                if robot_type.upper() in ["AGV", "AMR"]:
                    # Jetbot을 AGV 대용으로 사용 (Isaac Sim 기본 에셋)
                    robot_usd = assets_root + "/Isaac/Robots/Jetbot/jetbot.usd"
                    prim_path = f"/World/Robots/AGV_{robot_idx}"

                    add_reference_to_stage(robot_usd, prim_path)

                    robot = self.world.scene.add(
                        WheeledRobot(
                            prim_path=prim_path,
                            name=f"agv_{robot_idx}",
                            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                            create_robot=True,
                            position=np.array([x_pos, y_pos, 0.05])
                        )
                    )
                    robot._max_speed = speed
                    self.robots.append(robot)

                robot_idx += 1

        logger.info(f"Created {len(self.robots)} robots")

    async def run_simulation(self, duration: float = 60.0) -> Dict[str, Any]:
        """시뮬레이션 실행"""
        if not self.world:
            return {"success": False, "error": "World not initialized"}

        if self.is_running:
            return {"success": False, "error": "Simulation already running"}

        logger.info(f"Starting simulation for {duration} seconds")
        self.is_running = True
        self.metrics = {"throughput": 0, "collisions": 0, "items_processed": 0}

        time_step = 1/60
        total_steps = int(duration / time_step)
        render_interval = 3  # 매 3스텝마다 렌더링

        try:
            for step in range(total_steps):
                # 로봇 제어 (간단한 자율 주행 시뮬레이션)
                for robot in self.robots:
                    # 랜덤 이동 (실제로는 경로 계획 알고리즘 사용)
                    left_speed = np.random.uniform(0.3, robot._max_speed)
                    right_speed = np.random.uniform(0.3, robot._max_speed)

                    # 벽 회피 (간단한 로직)
                    pos = robot.get_world_pose()[0]
                    if abs(pos[0]) > 12 or abs(pos[1]) > 8:
                        # 중앙으로 돌아가기
                        left_speed, right_speed = right_speed, left_speed

                    robot.apply_wheel_actions(np.array([left_speed, right_speed]))

                # 물리 스텝 (렌더링은 간헐적으로)
                render = (step % render_interval == 0)
                self.world.step(render=render)

                # 메트릭 수집
                if step % 60 == 0:  # 1초마다
                    self.metrics["items_processed"] += len(self.robots)

                # 충돌 감지 (간략화)
                if step % 300 == 0 and np.random.random() < 0.05:
                    self.metrics["collisions"] += 1

                # 진행상황 로깅
                if step % (total_steps // 10) == 0:
                    progress = (step / total_steps) * 100
                    logger.info(f"Simulation progress: {progress:.0f}%")

            # 처리량 계산 (시간당)
            self.metrics["throughput"] = self.metrics["items_processed"] / (duration / 3600)

            result = {
                "success": True,
                "duration": duration,
                "total_throughput": round(self.metrics["throughput"], 2),
                "total_collisions": self.metrics["collisions"],
                "total_items_processed": self.metrics["items_processed"],
                "robot_count": len(self.robots)
            }
            logger.info(f"Simulation completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return {"success": False, "error": str(e)}

        finally:
            self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "initialized": self.world is not None,
            "is_running": self.is_running,
            "robots": len(self.robots),
            "obstacles": len(self.obstacles),
            "has_config": self.current_config is not None,
            "livestream_url": "http://localhost:8899/streaming/webrtc-client"
        }


# ============================================================
# WebSocket 서버
# ============================================================

manager = IsaacSimManager()


async def handle_message(websocket, message: str):
    """WebSocket 메시지 처리"""
    try:
        data = json.loads(message)
        msg_type = data.get("type", "")

        logger.info(f"Received: {msg_type}")

        if msg_type == "ping":
            return {"type": "pong", "status": "ok"}

        elif msg_type == "status":
            return {"type": "status", **manager.get_status()}

        elif msg_type == "command":
            command = data.get("command", "")
            params = data.get("params", {})

            if command == "initialize":
                result = await manager.initialize()
                return {"type": "result", "command": command, **result}

            elif command == "create_environment":
                result = await manager.create_environment(params)
                return {"type": "result", "command": command, **result}

            elif command == "run_simulation":
                duration = params.get("duration", 60.0)
                result = await manager.run_simulation(duration)
                return {"type": "result", "command": command, **result}

            elif command == "get_status":
                return {"type": "status", **manager.get_status()}

            else:
                return {"type": "error", "error": f"Unknown command: {command}"}

        else:
            return {"type": "error", "error": f"Unknown message type: {msg_type}"}

    except json.JSONDecodeError:
        return {"type": "error", "error": "Invalid JSON"}
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        return {"type": "error", "error": str(e)}


async def websocket_handler(websocket, path):
    """WebSocket 연결 핸들러"""
    client_addr = websocket.remote_address
    logger.info(f"Client connected: {client_addr}")

    try:
        async for message in websocket:
            response = await handle_message(websocket, message)
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


async def main():
    """메인 함수"""
    # World 초기화
    await manager.initialize()

    # WebSocket 서버 시작
    server = await websockets.serve(
        websocket_handler,
        "0.0.0.0",
        8211,
        ping_interval=30,
        ping_timeout=10
    )

    logger.info("Isaac Sim WebSocket Server started on port 8211")
    logger.info("Livestream available at http://localhost:8899/streaming/webrtc-client")

    # 시그널 핸들러
    stop = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

    # 메인 루프 (Isaac Sim 렌더링 유지)
    while not stop.is_set():
        simulation_app.update()
        await asyncio.sleep(1/30)  # 30 FPS

    # 정리
    server.close()
    await server.wait_closed()
    logger.info("Server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        simulation_app.close()
        logger.info("Isaac Sim closed")
