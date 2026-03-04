"""
Forge API Server v4.0 — NL 시뮬 자동화 + Sim2Real Gap 정량화
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
    title="Forge API v4.0",
    description="""
## NL 시뮬 자동화 + Sim2Real Gap 정량화 엔진

### 주요 기능
- **자연어 → Isaac Sim 환경 자동 구성**
- **DR 병렬 시뮬레이션 → 신뢰도 + 민감도 분석**
- **의사결정 리포트 생성**
- **민감도 기반 LLM 분기** (OpenAI / Qwen)
    """,
    version="4.0.0",
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
    prompt: str = Field(..., example="B동에 OHT 3대 추가하면 병목이 해소돼?")
    use_rag: bool = Field(default=True, description="RAG 사용 여부")
    provider: ProviderEnum = Field(default=ProviderEnum.auto, description="LLM 프로바이더 (auto=민감도 자동 분류)")
    security_mode: SecurityMode = Field(
        default=SecurityMode.secure,
        description="보안 모드: secure(로컬, 기본) / fast(외부 API 허용)"
    )


class ModifyRequest(BaseModel):
    """파라미터 수정 요청"""
    current_config: dict = Field(..., description="현재 설정")
    prompt: str = Field(..., example="OHT 2대 더 추가해줘")
    security_mode: SecurityMode = Field(
        default=SecurityMode.secure,
        description="보안 모드: secure(로컬) / fast(외부 API)"
    )


class SimulationResultRequest(BaseModel):
    """시뮬레이션 결과 분석 요청"""
    config: dict = Field(..., description="시뮬레이션 설정")
    result: dict = Field(..., description="시뮬레이션 결과")


class FeedbackRequest(BaseModel):
    """피드백 요청"""
    user_input: str
    llm_output: dict
    is_positive: bool
    comment: str = ""


class IsaacSimCommand(BaseModel):
    """Isaac Sim 명령"""
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
        "version": "4.0.0",
        "status": "running",
        "features": {
            "llm_providers": ["openai", "qwen"],
            "sensitivity_routing": True,
            "rag_enabled": True,
            "dr_analysis": True
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
    자연어 → Isaac Sim 환경 파라미터 변환

    - **prompt**: 자연어 요청 (예: "B동에 OHT 3대 추가하면 병목이 해소돼?")
    - **use_rag**: RAG 사용 여부 (문서/예시 참조)
    - **provider**: LLM 프로바이더 (auto/openai/ollama)
    - **security_mode**: 보안 모드
    """
    try:
        allow_external = request.security_mode == SecurityMode.fast

        force_provider = None
        if request.provider != ProviderEnum.auto:
            force_provider = LLMProvider(request.provider.value)

        if request.use_rag:
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
    """대화형 파라미터 수정"""
    try:
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


# ============================================================
# Isaac Sim 연동
# ============================================================

from isaac.client import get_client, IsaacSimConfig

_isaac_client = None


def get_isaac_client():
    """Isaac Sim 클라이언트 싱글톤"""
    global _isaac_client
    if _isaac_client is None:
        _isaac_client = get_client()
    return _isaac_client


@app.post("/isaac-sim/execute", tags=["Isaac Sim"])
async def execute_isaac_sim(command: IsaacSimCommand):
    """Isaac Sim 명령 실행"""
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

    return {"status": "error", "message": f"Unknown action: {command.action}"}


@app.get("/isaac-sim/status", tags=["Isaac Sim"])
async def isaac_sim_status():
    """Isaac Sim 연결 상태 확인"""
    client = get_isaac_client()

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
    """사용자 피드백 제출 (지속학습)"""
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
