#!/usr/bin/env python3
"""
Simple Isaac Sim rendering test using Replicator
For Docker: nvcr.io/nvidia/isaac-sim:2023.1.1
"""
import json
import base64
import sys

print("[1/6] Starting Isaac Sim...", file=sys.stderr)

# Docker 이미지용 import 방식
from omni.isaac.kit import SimulationApp

config = {
    "headless": True,
    "width": 1280,
    "height": 720,
}

simulation_app = SimulationApp(config)

print("[2/6] Imports...", file=sys.stderr)

import omni
import omni.replicator.core as rep
from omni.isaac.core import World
from pxr import Gf, UsdGeom, Usd
import numpy as np

try:
    print("[3/6] Creating scene...", file=sys.stderr)

    # Stage 가져오기
    stage = omni.usd.get_context().get_stage()

    # 바닥
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.25))
    UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3d(30, 20, 0.5))
    floor.CreateDisplayColorAttr([(0.4, 0.4, 0.4)])

    # 로봇들 (파란 큐브)
    for i in range(3):
        bot = UsdGeom.Cube.Define(stage, f"/World/AGV_{i}")
        UsdGeom.Xformable(bot).AddTranslateOp().Set(Gf.Vec3d(-8 + i * 8, 0, 0.75))
        UsdGeom.Xformable(bot).AddScaleOp().Set(Gf.Vec3d(1.5, 1.0, 1.5))
        bot.CreateDisplayColorAttr([(0.1, 0.3, 0.8)])

    # 선반 (갈색)
    for i in range(2):
        shelf = UsdGeom.Cube.Define(stage, f"/World/Shelf_{i}")
        y_pos = -6 if i == 0 else 6
        UsdGeom.Xformable(shelf).AddTranslateOp().Set(Gf.Vec3d(0, y_pos, 2))
        UsdGeom.Xformable(shelf).AddScaleOp().Set(Gf.Vec3d(20, 1.5, 4))
        shelf.CreateDisplayColorAttr([(0.55, 0.35, 0.15)])

    # 조명
    light = UsdGeom.DomeLight.Define(stage, "/World/DomeLight")
    light.CreateIntensityAttr(1000)

    print("[4/6] Setting up camera...", file=sys.stderr)

    # 카메라 (조감도)
    camera = rep.create.camera(
        position=(0, -30, 25),
        rotation=(55, 0, 0),
        focal_length=24
    )

    # Render Product
    rp = rep.create.render_product(camera, (1280, 720))

    print("[5/6] Rendering...", file=sys.stderr)

    # RGB annotator
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])

    # 여러 번 step해서 렌더링 안정화
    for _ in range(30):
        simulation_app.update()

    # Replicator step
    rep.orchestrator.step(rt_subframes=8)

    print("[6/6] Capturing...", file=sys.stderr)

    # 이미지 데이터 가져오기
    rgb_data = rgb_annot.get_data()

    if rgb_data is not None and len(rgb_data) > 0:
        # RGBA -> RGB (처음 3채널만)
        if len(rgb_data.shape) == 3 and rgb_data.shape[2] >= 3:
            rgb_array = rgb_data[:, :, :3]

            # PIL로 변환
            from PIL import Image
            import io

            img = Image.fromarray(rgb_array.astype(np.uint8))

            # JPEG로 인코딩
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            result = {
                "success": True,
                "image": img_base64,
                "width": img.width,
                "height": img.height,
                "format": "jpeg"
            }
            print(json.dumps(result))
        else:
            print(json.dumps({
                "success": False,
                "error": f"Invalid image shape: {rgb_data.shape}"
            }))
    else:
        print(json.dumps({
            "success": False,
            "error": "No image data captured"
        }))

except Exception as e:
    import traceback
    print(json.dumps({
        "success": False,
        "error": str(e),
        "trace": traceback.format_exc()
    }), file=sys.stderr)
    print(json.dumps({"success": False, "error": str(e)}))

finally:
    simulation_app.close()
