#!/usr/bin/env python3
"""
간단한 렌더링 테스트 스크립트
- 기본 큐브 생성
- 카메라로 캡처
- base64 이미지 출력
"""
import json
import base64
import io
import sys

print("Starting Isaac Sim...", file=sys.stderr)

from omni.isaac.kit import SimulationApp

config = {
    "headless": True,
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(config)

print("SimulationApp initialized", file=sys.stderr)

import omni
from omni.isaac.core import World
from pxr import Gf, UsdGeom
import numpy as np

try:
    # 월드 생성
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    print("Creating scene...", file=sys.stderr)

    # 바닥 (평면)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor_xform = UsdGeom.Xformable(floor)
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
    floor_xform.AddScaleOp().Set(Gf.Vec3d(30, 20, 0.1))
    floor.CreateDisplayColorAttr([(0.3, 0.3, 0.3)])

    # 로봇 (파란 큐브)
    for i in range(3):
        robot = UsdGeom.Cube.Define(stage, f"/World/Robot_{i}")
        robot_xform = UsdGeom.Xformable(robot)
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(-10 + i * 10, 0, 0.5))
        robot_xform.AddScaleOp().Set(Gf.Vec3d(1, 1, 1))
        robot.CreateDisplayColorAttr([(0.2, 0.4, 0.9)])

    # 선반 (갈색 박스)
    for i in range(2):
        shelf = UsdGeom.Cube.Define(stage, f"/World/Shelf_{i}")
        shelf_xform = UsdGeom.Xformable(shelf)
        shelf_xform.AddTranslateOp().Set(Gf.Vec3d(0, -5 + i * 10, 1.5))
        shelf_xform.AddScaleOp().Set(Gf.Vec3d(20, 1, 3))
        shelf.CreateDisplayColorAttr([(0.6, 0.4, 0.2)])

    # 조명
    light = UsdGeom.SphereLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(30000)
    light_xform = UsdGeom.Xformable(light)
    light_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 20))

    # 카메라
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0, -25, 20))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3d(60, 0, 0))
    camera.CreateFocalLengthAttr(24.0)

    print("Scene created, initializing world...", file=sys.stderr)

    # 월드 초기화
    world.reset()

    # 렌더링 (안정화를 위해 여러 프레임)
    print("Rendering frames...", file=sys.stderr)
    for i in range(30):
        world.step(render=True)

    print("Capturing image...", file=sys.stderr)

    # Viewport에서 이미지 캡처
    import omni.kit.viewport.utility as vp_utils
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport:
        viewport.set_active_camera("/World/Camera")

        # 렌더링 대기
        for _ in range(10):
            world.step(render=True)

        # 캡처
        import omni.kit.capture.viewport as capture

        # 파일로 저장
        capture_path = "/tmp/render_output.png"

        # 캡처 대기
        import asyncio

        async def capture_frame():
            cap = capture.CaptureExtension.get_instance()
            cap.options.output_folder = "/tmp"
            cap.options.file_name = "render_output"
            cap.options.file_type = capture.FileType.PNG
            await cap.capture_next_frame()

        asyncio.get_event_loop().run_until_complete(capture_frame())

        # 파일 읽기
        import time
        time.sleep(1)

        with open(capture_path, "rb") as f:
            img_data = f.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        result = {
            "success": True,
            "image": img_base64,
            "format": "png"
        }
    else:
        result = {
            "success": False,
            "error": "No viewport available"
        }

    print(json.dumps(result))

except Exception as e:
    import traceback
    result = {
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }
    print(json.dumps(result))

finally:
    simulation_app.close()
