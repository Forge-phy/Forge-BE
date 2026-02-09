"""
Forge API Extension for Isaac Sim
- HTTP API 서버 제공
- 환경 생성 / 시뮬레이션 실행 / 카메라 캡처
"""
import omni.ext
import omni.kit.app
import omni.usd
import carb

from pxr import Gf, UsdGeom
import asyncio
import json
import base64
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import numpy as np

# Global references
_extension_instance = None
_server = None


class ForgeAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler"""

    def log_message(self, format, *args):
        carb.log_info(f"[Forge API] {args[0]}")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/status':
            self._send_json({
                "status": "running",
                "extension": "forge.api",
                "version": "1.0.0"
            })
        elif self.path == '/capture':
            # 카메라 캡처
            result = _extension_instance.capture_viewport()
            self._send_json(result)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body) if body else {}
        except:
            data = {}

        if self.path == '/create_environment':
            result = _extension_instance.create_environment(data)
            self._send_json(result)
        elif self.path == '/run_simulation':
            result = _extension_instance.run_simulation(data)
            self._send_json(result)
        elif self.path == '/capture':
            result = _extension_instance.capture_viewport()
            self._send_json(result)
        else:
            self._send_json({"error": "Not found"}, 404)


class ForgeAPIExtension(omni.ext.IExt):
    """Forge API Extension"""

    def on_startup(self, ext_id):
        global _extension_instance, _server
        _extension_instance = self

        carb.log_info("[Forge API] Extension starting...")

        # HTTP 서버 시작
        self._start_server(port=8080)

        carb.log_info("[Forge API] Extension started on port 8080")

    def on_shutdown(self):
        global _server
        carb.log_info("[Forge API] Extension shutting down...")

        if _server:
            _server.shutdown()
            _server = None

    def _start_server(self, port=8080):
        global _server

        def run_server():
            global _server
            _server = HTTPServer(('0.0.0.0', port), ForgeAPIHandler)
            carb.log_info(f"[Forge API] HTTP server running on port {port}")
            _server.serve_forever()

        thread = Thread(target=run_server, daemon=True)
        thread.start()

    def create_environment(self, params: dict) -> dict:
        """환경 생성"""
        try:
            stage = omni.usd.get_context().get_stage()

            # 기존 환경 정리
            world_prim = stage.GetPrimAtPath("/World")
            if world_prim.IsValid():
                for child in world_prim.GetChildren():
                    stage.RemovePrim(child.GetPath())

            # 파라미터 추출
            env = params.get("environment", {})
            width = env.get("width", 30)
            length = env.get("length", 20)

            robots = params.get("robots", [{"type": "AGV", "count": 3}])
            shelves = params.get("shelves", [{"rows": 2}])

            created_objects = []

            # 바닥 생성
            floor = UsdGeom.Cube.Define(stage, "/World/Floor")
            UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.25))
            UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3d(width, length, 0.5))
            floor.CreateDisplayColorAttr([(0.4, 0.4, 0.4)])
            created_objects.append("/World/Floor")

            # 로봇 생성
            robot_idx = 0
            for robot_config in robots:
                count = robot_config.get("count", 1)
                for i in range(count):
                    path = f"/World/Robot_{robot_idx}"
                    robot = UsdGeom.Cube.Define(stage, path)

                    # 위치 계산
                    x = -width/2 + (i + 1) * width / (count + 1)
                    y = 0

                    UsdGeom.Xformable(robot).AddTranslateOp().Set(Gf.Vec3d(x, y, 0.75))
                    UsdGeom.Xformable(robot).AddScaleOp().Set(Gf.Vec3d(1.5, 1.0, 1.5))
                    robot.CreateDisplayColorAttr([(0.1, 0.3, 0.8)])

                    created_objects.append(path)
                    robot_idx += 1

            # 선반 생성
            for i, shelf_config in enumerate(shelves):
                rows = shelf_config.get("rows", 2)
                for row in range(rows):
                    path = f"/World/Shelf_{i}_{row}"
                    shelf = UsdGeom.Cube.Define(stage, path)

                    y = -length/2 + (row + 1) * length / (rows + 1)

                    UsdGeom.Xformable(shelf).AddTranslateOp().Set(Gf.Vec3d(0, y, 2))
                    UsdGeom.Xformable(shelf).AddScaleOp().Set(Gf.Vec3d(width * 0.8, 1.5, 4))
                    shelf.CreateDisplayColorAttr([(0.55, 0.35, 0.15)])

                    created_objects.append(path)

            # 조명
            light = UsdGeom.DomeLight.Define(stage, "/World/Light")
            light.CreateIntensityAttr(1000)
            created_objects.append("/World/Light")

            # 카메라
            camera = UsdGeom.Camera.Define(stage, "/World/Camera")
            cam_xform = UsdGeom.Xformable(camera)
            cam_xform.AddTranslateOp().Set(Gf.Vec3d(0, -length, length/2))
            cam_xform.AddRotateXYZOp().Set(Gf.Vec3d(45, 0, 0))
            created_objects.append("/World/Camera")

            carb.log_info(f"[Forge API] Created environment with {len(created_objects)} objects")

            return {
                "success": True,
                "message": f"Environment created with {len(created_objects)} objects",
                "objects": created_objects,
                "dimensions": {"width": width, "length": length}
            }

        except Exception as e:
            carb.log_error(f"[Forge API] Error creating environment: {e}")
            return {"success": False, "error": str(e)}

    def run_simulation(self, params: dict) -> dict:
        """시뮬레이션 실행"""
        try:
            duration = params.get("duration", 10)  # seconds

            # 시뮬레이션 시작
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()

            carb.log_info(f"[Forge API] Simulation started for {duration}s")

            return {
                "success": True,
                "message": f"Simulation started for {duration} seconds",
                "status": "running"
            }

        except Exception as e:
            carb.log_error(f"[Forge API] Error running simulation: {e}")
            return {"success": False, "error": str(e)}

    def capture_viewport(self) -> dict:
        """뷰포트 캡처"""
        try:
            import omni.replicator.core as rep

            # 카메라로부터 렌더링
            camera_path = "/World/Camera"

            # Render Product 생성
            rp = rep.create.render_product(camera_path, (1280, 720))

            # RGB annotator
            rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annot.attach([rp])

            # 렌더링
            rep.orchestrator.step(rt_subframes=4)

            # 이미지 가져오기
            rgb_data = rgb_annot.get_data()

            if rgb_data is not None and len(rgb_data) > 0:
                from PIL import Image

                rgb_array = rgb_data[:, :, :3]
                img = Image.fromarray(rgb_array.astype(np.uint8))

                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return {
                    "success": True,
                    "image": img_base64,
                    "width": img.width,
                    "height": img.height,
                    "format": "jpeg"
                }
            else:
                return {"success": False, "error": "No image data"}

        except Exception as e:
            carb.log_error(f"[Forge API] Error capturing: {e}")
            return {"success": False, "error": str(e)}
