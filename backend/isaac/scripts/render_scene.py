#!/usr/bin/env python3
"""
Isaac Sim 환경 렌더링 스크립트
- 환경 파라미터를 받아서 씬 생성
- 카메라로 렌더링
- 이미지를 base64로 출력
"""
import sys
import json
import base64
import io

# Isaac Sim imports
from omni.isaac.kit import SimulationApp

# Headless 모드로 시작
config = {
    "headless": True,
    "width": 1280,
    "height": 720,
    "anti_aliasing": 1,
    "renderer": "RayTracedLighting"
}

simulation_app = SimulationApp(config)

# 나머지 import는 SimulationApp 초기화 후에
import omni
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom, Sdf
import numpy as np
from PIL import Image

import omni.replicator.core as rep


def create_warehouse_scene(params: dict):
    """창고 환경 생성"""
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # 파라미터 추출
    env = params.get("environment", {})
    width = env.get("width", 30)
    length = env.get("length", 20)

    robots = params.get("robots", [])
    shelves = params.get("shelves", [])

    # 1. 바닥 생성
    floor_path = "/World/Floor"
    floor = UsdGeom.Mesh.Define(stage, floor_path)

    # 간단한 평면
    floor.CreatePointsAttr([
        Gf.Vec3f(0, 0, 0),
        Gf.Vec3f(width, 0, 0),
        Gf.Vec3f(width, length, 0),
        Gf.Vec3f(0, length, 0)
    ])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    # 바닥 색상 (회색)
    floor.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])

    # 2. 벽 생성 (선택적)
    wall_height = 5.0

    # 3. AGV 로봇 생성 (간단한 박스로 표현)
    for i, robot in enumerate(robots):
        robot_type = robot.get("type", "AGV")
        count = robot.get("count", 1)

        for j in range(count):
            robot_path = f"/World/Robots/Robot_{i}_{j}"

            # 위치 계산 (균등 분배)
            x = (j + 1) * width / (count + 1)
            y = length / 2

            # 큐브로 로봇 표현
            cube = UsdGeom.Cube.Define(stage, robot_path)
            cube.CreateSizeAttr(1.0)

            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.5))

            # 로봇 색상 (파란색)
            cube.CreateDisplayColorAttr([(0.2, 0.4, 0.8)])

    # 4. 선반 생성
    for i, shelf in enumerate(shelves):
        rows = shelf.get("rows", 2)

        for row in range(rows):
            shelf_path = f"/World/Shelves/Shelf_{i}_{row}"

            # 위치 계산
            x = width / 2
            y = (row + 1) * length / (rows + 1)

            # 선반 (긴 박스)
            cube = UsdGeom.Cube.Define(stage, shelf_path)

            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, 1.5))
            xform.AddScaleOp().Set(Gf.Vec3d(width * 0.8, 1, 3))

            # 선반 색상 (갈색)
            cube.CreateDisplayColorAttr([(0.6, 0.4, 0.2)])

    return world, stage


def setup_camera(stage, width, length):
    """카메라 설정 (조감도)"""
    camera_path = "/World/Camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)

    # 카메라 위치 (위에서 내려다보기)
    xform = UsdGeom.Xformable(camera)
    xform.AddTranslateOp().Set(Gf.Vec3d(width/2, length/2, 30))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))  # 아래를 향함

    # 카메라 속성
    camera.CreateFocalLengthAttr(24.0)

    return camera_path


def capture_image(camera_path: str, output_size=(1280, 720)):
    """카메라에서 이미지 캡처"""
    # Replicator를 사용한 렌더링
    rp = rep.create.render_product(camera_path, output_size)

    # RGB 출력
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    # 렌더링 실행
    rep.orchestrator.step()

    # 이미지 데이터 가져오기
    data = rgb.get_data()

    if data is not None:
        # numpy array를 PIL Image로 변환
        img = Image.fromarray(data[:, :, :3].astype(np.uint8))
        return img

    return None


def main():
    """메인 함수"""
    try:
        # stdin에서 파라미터 읽기
        if len(sys.argv) > 1:
            params = json.loads(sys.argv[1])
        else:
            params_str = sys.stdin.read()
            params = json.loads(params_str) if params_str else {}

        # 기본값
        if not params:
            params = {
                "environment": {"width": 30, "length": 20},
                "robots": [{"type": "AGV", "count": 3}],
                "shelves": [{"rows": 2}]
            }

        # 씬 생성
        world, stage = create_warehouse_scene(params)

        # 환경 크기
        env = params.get("environment", {})
        width = env.get("width", 30)
        length = env.get("length", 20)

        # 카메라 설정
        camera_path = setup_camera(stage, width, length)

        # 월드 초기화
        world.reset()

        # 몇 프레임 렌더링 (안정화)
        for _ in range(10):
            world.step(render=True)

        # 이미지 캡처
        img = capture_image(camera_path)

        if img:
            # 이미지를 base64로 인코딩
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            result = {
                "success": True,
                "image": img_base64,
                "width": img.width,
                "height": img.height,
                "format": "jpeg"
            }
        else:
            result = {
                "success": False,
                "error": "Failed to capture image"
            }

        print(json.dumps(result))

    except Exception as e:
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result))

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
