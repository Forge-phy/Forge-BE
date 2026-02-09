"""
Forge API Server v2.1 - 보안 우선 + OpenAI/Qwen 하이브리드
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')

# .env 로드
from dotenv import load_dotenv
load_dotenv()

from llm.providers import init_config, LLMProvider, SensitivityLevel
from llm.service import (
    parse_environment_request,
    modify_parameters,
    analyze_simulation_result,
    generate_report_from_lstm,
    LLMResponse
)
from rag import (
    rag_parse_environment,
    rag_analyze_result,
    submit_feedback,
    manually_add_good_example,
    get_collection_count
)

# LLM 설정 초기화
init_config()

app = FastAPI(
    title="Forge API v2.0",
    description="""
## Isaac Sim 브릿지 + 공장 환경 예측 + 지속학습 API

### 주요 기능
- **자연어 → Isaac Sim 파라미터 변환**
- **시뮬레이션 결과 분석**
- **LSTM → LLM 보고서 생성**
- **민감도 기반 LLM 분기** (OpenAI / Qwen)

### 민감도 분류
- `public`: 일반 정보 → OpenAI (빠름)
- `internal`: 내부 정보 → Qwen 로컬 (보안)
- `confidential`: 기밀 정보 → Qwen 로컬 (보안)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class ProviderEnum(str, Enum):
    openai = "openai"
    ollama = "ollama"
    auto = "auto"


class SecurityMode(str, Enum):
    """보안 모드"""
    secure = "secure"    # 로컬 처리 (기본값, 데이터 외부 전송 없음)
    fast = "fast"        # 외부 API 허용 (빠른 응답)


class EnvironmentRequest(BaseModel):
    """환경 설정 요청"""
    prompt: str = Field(..., example="30m x 20m 창고에 AGV 3대, 선반 2열")
    use_rag: bool = Field(default=True, description="RAG 사용 여부")
    provider: ProviderEnum = Field(default=ProviderEnum.auto, description="LLM 프로바이더 (auto=민감도 자동 분류)")
    security_mode: SecurityMode = Field(
        default=SecurityMode.secure,
        description="보안 모드: secure(로컬, 기본) / fast(외부 API 허용)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "30m x 20m 창고에 AGV 3대, 선반 2열",
                "use_rag": True,
                "provider": "auto",
                "security_mode": "secure"
            }
        }


class ModifyRequest(BaseModel):
    """파라미터 수정 요청"""
    current_config: dict = Field(..., description="현재 설정")
    prompt: str = Field(..., example="속도 올려줘")
    security_mode: SecurityMode = Field(
        default=SecurityMode.secure,
        description="보안 모드: secure(로컬) / fast(외부 API)"
    )


class SimulationResultRequest(BaseModel):
    """시뮬레이션 결과 분석 요청"""
    config: dict = Field(..., description="시뮬레이션 설정")
    result: dict = Field(..., description="시뮬레이션 결과")


class LSTMResultRequest(BaseModel):
    """LSTM 예측 결과 → 보고서 생성 요청"""
    lstm_result: dict = Field(..., description="LSTM 예측 결과")
    sim_config: dict = Field(..., description="시뮬레이션 설정")

    class Config:
        json_schema_extra = {
            "example": {
                "lstm_result": {
                    "sim_throughput": 100,
                    "predicted_real_throughput": 85,
                    "gap_percent": 15,
                    "confidence": 0.89,
                    "factors": ["temperature", "humidity"],
                    "confidence_interval": [83, 87]
                },
                "sim_config": {
                    "environment": {"type": "warehouse", "width": 30, "length": 20},
                    "robots": [{"type": "AGV", "count": 3}]
                }
            }
        }


class FeedbackRequest(BaseModel):
    """피드백 요청"""
    user_input: str
    llm_output: dict
    is_positive: bool
    comment: str = ""


class IsaacSimCommand(BaseModel):
    """Isaac Sim 명령 (향후 연동용)"""
    action: str = Field(..., example="create_environment")
    parameters: dict = Field(..., description="Isaac Sim 파라미터")


# ============================================================
# 세션 저장
# ============================================================
current_session = {"config": None, "last_input": None}


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", tags=["Status"])
async def root():
    """API 상태 확인"""
    return {
        "service": "Forge API",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "llm_providers": ["openai", "qwen"],
            "sensitivity_routing": True,
            "rag_enabled": True,
            "lstm_report_generation": True
        },
        "stats": {
            "documents": get_collection_count("documents"),
            "examples": get_collection_count("examples"),
            "feedback": get_collection_count("feedback")
        }
    }


@app.post("/llm/environment", tags=["LLM"])
async def create_environment(request: EnvironmentRequest):
    """
    자연어 → Isaac Sim 환경 파라미터 변환 (v2.1 보안 우선)

    - **prompt**: 자연어 요청 (예: "30m x 20m 창고에 AGV 3대")
    - **use_rag**: RAG 사용 여부 (문서/예시 참조)
    - **provider**: LLM 프로바이더 (auto/openai/ollama)
    - **security_mode**: 보안 모드
      - `secure` (기본): 로컬 처리, 데이터 외부 전송 없음
      - `fast`: 외부 API 허용, 빠른 응답
    """
    try:
        # v2.1: 보안 모드 확인
        allow_external = request.security_mode == SecurityMode.fast

        # 프로바이더 설정
        force_provider = None
        if request.provider != ProviderEnum.auto:
            force_provider = LLMProvider(request.provider.value)

        # LLM 호출
        if request.use_rag:
            # RAG v2.1: 보안 우선 모드 적용
            rag_response = await rag_parse_environment(
                request.prompt,
                allow_external=allow_external
            )
            result = rag_response.data
            provider_used = rag_response.provider
            sensitivity = rag_response.sensitivity
            sensitivity_reason = rag_response.reason
        else:
            response = await parse_environment_request(
                request.prompt,
                force_provider=force_provider,
                allow_external=allow_external
            )
            result = response.data
            provider_used = response.provider
            sensitivity = response.sensitivity
            sensitivity_reason = ""

        if not result:
            raise HTTPException(status_code=400, detail="파라미터 생성 실패")

        current_session["config"] = result
        current_session["last_input"] = request.prompt

        response_data = {
            "status": "success",
            "provider": provider_used,
            "sensitivity": sensitivity,
            "security_mode": request.security_mode.value,
            "parameters": result,
            "message": "환경 파라미터가 생성되었습니다"
        }

        # RAG 사용시 민감도 분류 상세 정보 추가
        if request.use_rag and sensitivity_reason:
            response_data["sensitivity_detail"] = {
                "reason": sensitivity_reason,
                "input_sensitivity": rag_response.input_sensitivity,
                "context_sensitivity": rag_response.context_sensitivity
            }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/modify", tags=["LLM"])
async def modify_config(request: ModifyRequest):
    """
    대화형 파라미터 수정 (v2.1 보안 모드 지원)

    예시: "속도 올려줘", "로봇 2대 더 추가"
    """
    try:
        # v2.1: 보안 모드 확인
        allow_external = request.security_mode == SecurityMode.fast

        response = await modify_parameters(
            request.current_config,
            request.prompt,
            allow_external=allow_external
        )

        if not response.success:
            raise HTTPException(status_code=400, detail=response.error)

        current_session["config"] = response.data

        return {
            "status": "success",
            "provider": response.provider,
            "sensitivity": response.sensitivity,
            "security_mode": request.security_mode.value,
            "parameters": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/analyze", tags=["LLM"])
async def analyze_result(request: SimulationResultRequest):
    """시뮬레이션 결과 분석"""
    try:
        response = await analyze_simulation_result(request.config, request.result)

        return {
            "status": "success",
            "provider": response.provider,
            "sensitivity": response.sensitivity,
            "analysis": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/report", tags=["LLM", "LSTM"])
async def generate_report(request: LSTMResultRequest):
    """
    LSTM 예측 결과 → 자연어 보고서 생성 (v2.0 핵심)

    LSTM의 숫자 예측 결과를 경영진이 이해할 수 있는 보고서로 변환
    """
    try:
        response = await generate_report_from_lstm(
            request.lstm_result,
            request.sim_config
        )

        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return {
            "status": "success",
            "provider": response.provider,
            "sensitivity": response.sensitivity,
            "report": response.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Isaac Sim 연동 (실제 클라이언트 사용)
# ============================================================

from isaac.client import get_client, IsaacSimConfig

# Isaac Sim 클라이언트 (지연 초기화)
_isaac_client = None


def get_isaac_client():
    """Isaac Sim 클라이언트 싱글톤"""
    global _isaac_client
    if _isaac_client is None:
        _isaac_client = get_client()
    return _isaac_client


@app.post("/isaac-sim/execute", tags=["Isaac Sim"])
async def execute_isaac_sim(command: IsaacSimCommand):
    """
    Isaac Sim 명령 실행

    지원 액션:
    - create_environment: 환경 생성
    - run_simulation: 시뮬레이션 실행
    - get_status: 상태 확인
    """
    client = get_isaac_client()

    if command.action == "create_environment":
        result = await client.create_environment(command.parameters)
        return {
            "status": "success" if result.get("success") else "error",
            "message": result.get("message", "환경이 생성되었습니다"),
            "scene_id": result.get("scene_id"),
            "objects_created": result.get("objects_created", 0),
            "stream_url": client.stream_url
        }

    elif command.action == "run_simulation":
        duration = command.parameters.get("duration", 3600)
        result = await client.run_simulation(duration)
        return {
            "status": "success" if result.get("success") else "error",
            "message": "시뮬레이션이 완료되었습니다",
            "duration": result.get("duration"),
            "result": {
                "throughput": result.get("total_throughput"),
                "collisions": result.get("total_collisions"),
                "items_processed": result.get("total_items_processed")
            }
        }

    elif command.action == "get_status":
        status = client.get_status()
        return {
            "status": "success",
            "isaac_sim_connected": client.is_connected(),
            "mode": status.get("mode"),
            "stream_url": client.stream_url,
            "message": f"Isaac Sim {status.get('mode')} 모드"
        }

    elif command.action == "start_simulation":
        # Isaac Sim Forge API 직접 호출 (로봇 움직임 시작)
        import aiohttp
        forge_api_url = f"http://{client.config.host}:{client.config.stream_port}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{forge_api_url}/forge/start-simulation",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    return {
                        "success": result.get("success", False),
                        "message": result.get("message", "시뮬레이션 시작"),
                        "robots": result.get("robots", 0)
                    }
        except Exception as e:
            return {"success": False, "message": f"연결 실패: {e}"}

    elif command.action == "stop_simulation":
        # Isaac Sim Forge API 직접 호출 (시뮬레이션 정지)
        import aiohttp
        forge_api_url = f"http://{client.config.host}:{client.config.stream_port}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{forge_api_url}/forge/stop-simulation",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    return {
                        "success": result.get("success", False),
                        "message": result.get("message", "시뮬레이션 정지")
                    }
        except Exception as e:
            return {"success": False, "message": f"연결 실패: {e}"}

    return {"status": "error", "message": f"Unknown action: {command.action}"}


@app.get("/isaac-sim/status", tags=["Isaac Sim"])
async def isaac_sim_status():
    """Isaac Sim 연결 상태 확인"""
    client = get_isaac_client()

    # 연결 시도
    if not client.is_connected():
        await client.connect()

    status = client.get_status()

    return {
        "connected": client.is_connected(),
        "mode": status.get("mode", "mock"),
        "stream_url": client.stream_url,
        "message": f"Isaac Sim {status.get('mode')} 모드로 연결됨" if client.is_connected() else "연결 안됨",
        "details": status
    }


@app.get("/isaac-sim/stream", tags=["Isaac Sim"])
async def isaac_sim_stream_url():
    """Isaac Sim 스트리밍 URL 반환"""
    client = get_isaac_client()
    return {
        "stream_url": client.stream_url,
        "mode": client.mode,
        "available": client.stream_url is not None
    }


# ============================================================
# 지속학습 API
# ============================================================

@app.post("/feedback", tags=["Learning"])
async def submit_user_feedback(request: FeedbackRequest):
    """
    사용자 피드백 제출 (지속학습)

    - **is_positive=True**: 좋은 예시로 저장 → 향후 유사 질문에 참조
    - **is_positive=False**: 피드백만 저장
    """
    try:
        doc_id = submit_feedback(
            request.user_input,
            request.llm_output,
            request.is_positive,
            request.comment
        )
        return {
            "status": "success",
            "message": "피드백이 저장되었습니다" + (" (좋은 예시로 추가됨)" if request.is_positive else ""),
            "feedback_id": doc_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/quick", tags=["Learning"])
async def quick_feedback(is_positive: bool = Query(..., description="좋아요 여부")):
    """마지막 응답에 대한 빠른 피드백"""
    if not current_session["config"] or not current_session["last_input"]:
        raise HTTPException(status_code=400, detail="피드백할 이전 응답이 없습니다")

    try:
        doc_id = submit_feedback(
            current_session["last_input"],
            current_session["config"],
            is_positive,
            "quick_feedback"
        )
        return {
            "status": "success",
            "message": "좋아요!" if is_positive else "피드백 저장됨",
            "learned": is_positive
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Status"])
async def get_stats():
    """학습 현황 통계"""
    return {
        "documents": get_collection_count("documents"),
        "good_examples": get_collection_count("examples"),
        "feedback_count": get_collection_count("feedback"),
        "current_session": {
            "has_config": current_session["config"] is not None,
            "last_input": current_session["last_input"]
        }
    }


# ============================================================
# Pipeline API (Isaac Sim 통합 파이프라인)
# ============================================================

class PipelineRequest(BaseModel):
    """파이프라인 실행 요청"""
    prompt: str = Field(..., example="30m x 20m 창고에 AGV 3대, 온도 32도, 습도 70%")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "30m x 20m 창고에 AGV 3대, 온도 32도, 습도 70% 환경에서 1시간 시뮬레이션"
            }
        }


class PipelineConfigRequest(BaseModel):
    """설정 기반 파이프라인 실행 요청"""
    environment: dict = Field(..., description="환경 설정")
    simulation: Optional[dict] = Field(default=None, description="시뮬레이션 설정")

    class Config:
        json_schema_extra = {
            "example": {
                "environment": {
                    "warehouse": {"width": 30, "length": 20},
                    "robots": [{"type": "AGV", "count": 3}],
                    "temperature": 32,
                    "humidity": 70
                },
                "simulation": {
                    "duration": 3600,
                    "time_step": 0.1
                }
            }
        }


@app.post("/pipeline/run", tags=["Pipeline"])
async def run_pipeline(request: PipelineRequest):
    """
    자연어로 전체 파이프라인 실행

    전체 워크플로우:
    1. 자연어 파싱 (LLM)
    2. 환경 생성 (Isaac Sim)
    3. 시뮬레이션 실행
    4. Gap 예측 (LSTM)
    5. 보고서 생성 (LLM)
    """
    try:
        from isaac import IsaacSimPipeline

        pipeline = IsaacSimPipeline()
        result = await pipeline.run_from_natural_language(request.prompt)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        return {
            "status": "success",
            "result": result.to_dict()
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline 모듈 로드 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/config", tags=["Pipeline"])
async def run_pipeline_from_config(request: PipelineConfigRequest):
    """
    설정 객체로 파이프라인 실행

    자연어 파싱 단계를 건너뛰고 직접 설정으로 실행
    """
    try:
        from isaac import IsaacSimPipeline, EnvironmentConfig, SimulationConfig

        # 환경 설정 변환
        env_config = EnvironmentConfig.from_dict(request.environment)

        # 시뮬레이션 설정 변환
        sim_config = None
        if request.simulation:
            sim_config = SimulationConfig.from_dict(request.simulation)

        pipeline = IsaacSimPipeline()
        result = await pipeline.run_from_config(env_config, sim_config)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        return {
            "status": "success",
            "result": result.to_dict()
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline 모듈 로드 실패: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LSTM API (Sim2Real Gap 예측)
# ============================================================

class LSTMPredictRequest(BaseModel):
    """LSTM 예측 요청"""
    sim_throughput: float = Field(..., description="시뮬레이션 처리량 (개/시간)")
    temperature: float = Field(default=25.0, description="현장 온도 (°C)")
    humidity: float = Field(default=50.0, description="현장 습도 (%)")
    robot_count: int = Field(default=3, description="로봇 수")
    operation_hours: float = Field(default=4.0, description="연속 가동 시간")
    warehouse_size: float = Field(default=500.0, description="창고 크기 (m²)")

    class Config:
        json_schema_extra = {
            "example": {
                "sim_throughput": 100,
                "temperature": 32,
                "humidity": 75,
                "robot_count": 5,
                "operation_hours": 6,
                "warehouse_size": 600
            }
        }


# LSTM 예측기 인스턴스 (지연 로딩)
_lstm_predictor = None


def get_lstm_predictor():
    """LSTM 예측기 가져오기 (싱글톤) - v3 모델 우선"""
    global _lstm_predictor
    if _lstm_predictor is None:
        try:
            from lstm import Sim2RealPredictor
            from pathlib import Path

            _lstm_predictor = Sim2RealPredictor(model_type="lstm")

            # v3 모델 경로 (우선)
            v3_model_path = Path(__file__).parent.parent / "lstm" / "data" / "sim2real_lstm_v3.pth"
            # 기존 모델 경로 (fallback)
            legacy_model_path = Path(__file__).parent.parent / "lstm" / "checkpoints" / "mlp_sim2real_best.pt"

            if v3_model_path.exists():
                _lstm_predictor.load_model(str(v3_model_path))
                print(f"✅ LSTM v3 모델 로드: {v3_model_path}")
            elif legacy_model_path.exists():
                _lstm_predictor.load_model(str(legacy_model_path))
                print(f"✅ LSTM legacy 모델 로드: {legacy_model_path}")
            else:
                print(f"⚠️ LSTM 모델 파일 없음")
                _lstm_predictor = None

        except Exception as e:
            print(f"⚠️ LSTM 모듈 로드 실패: {e}")
            _lstm_predictor = None

    return _lstm_predictor


@app.post("/lstm/predict", tags=["LSTM"])
async def lstm_predict(request: LSTMPredictRequest):
    """
    LSTM Sim2Real Gap 예측

    시뮬레이션 결과와 환경 조건으로 실제 현장 성능을 예측합니다.
    """
    predictor = get_lstm_predictor()

    if predictor is None:
        # Fallback: 규칙 기반 예측
        gap_percent = -10.0
        temp_effect = -0.5 * abs(request.temperature - 25)
        humidity_effect = -0.3 * max(0, request.humidity - 60)
        congestion_effect = -1.0 * max(0, request.robot_count - 4)

        gap_percent += temp_effect + humidity_effect + congestion_effect
        gap_percent = max(-35, min(5, gap_percent))

        predicted_real = request.sim_throughput * (1 + gap_percent / 100)
        confidence = 0.6

        factors = []
        if request.temperature > 30:
            factors.append(f"고온 ({request.temperature}°C)")
        if request.humidity > 70:
            factors.append(f"고습도 ({request.humidity}%)")
        if request.robot_count > 6:
            factors.append(f"로봇 혼잡 ({request.robot_count}대)")
        if not factors:
            factors.append("정상 조건")

        return {
            "status": "success",
            "mode": "fallback",
            "prediction": {
                "sim_throughput": request.sim_throughput,
                "predicted_real_throughput": round(predicted_real, 2),
                "gap_percent": round(gap_percent, 2),
                "confidence": confidence,
                "confidence_interval": [
                    round(predicted_real * 0.97, 1),
                    round(predicted_real * 1.03, 1)
                ],
                "factors": factors
            }
        }

    try:
        result = predictor.predict(
            sim_throughput=request.sim_throughput,
            temperature=request.temperature,
            humidity=request.humidity,
            robot_count=request.robot_count,
            operation_hours=request.operation_hours,
            warehouse_size=request.warehouse_size,
            return_dict=True
        )

        return {
            "status": "success",
            "mode": "lstm",
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lstm/status", tags=["LSTM"])
async def lstm_status():
    """LSTM 모델 상태 확인"""
    predictor = get_lstm_predictor()

    from pathlib import Path
    model_path = Path(__file__).parent.parent / "lstm" / "checkpoints" / "mlp_sim2real_best.pt"

    return {
        "model_loaded": predictor is not None,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "mode": "lstm" if predictor else "fallback",
        "features": [
            "sim_throughput", "temperature", "humidity",
            "robot_count", "operation_hours", "warehouse_size"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
