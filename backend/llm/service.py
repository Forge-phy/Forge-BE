"""
Forge LLM Service v4.0 - OpenAI + Qwen 하이브리드
민감도에 따른 LLM 분기 처리
"""
import json
import re
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .providers import (
    call_llm,
    call_openai,
    call_ollama,
    extract_json,
    classify_sensitivity,
    select_provider,
    init_config,
    get_config,
    LLMProvider,
    SensitivityLevel
)
from .prompts import (
    SYSTEM_PROMPT,
    build_env_prompt,
    build_modify_prompt,
    build_analysis_prompt,
    build_error_prompt,
    build_comparison_prompt
)
from .schemas import (
    IsaacSimParameters,
    validate_and_normalize,
    SimulationResult as SimResultSchema
)
from .conversation import (
    ConversationManager,
    MessageRole,
    session_store
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forge.llm")


class LLMError(Exception):
    """LLM 관련 에러"""
    pass


@dataclass
class LLMResponse:
    """LLM 응답 래퍼"""
    success: bool
    data: Any
    raw_response: str
    provider: Optional[str] = None  # v2.0: 사용된 프로바이더
    sensitivity: Optional[str] = None  # v2.0: 민감도 레벨
    error: Optional[str] = None
    validation_errors: Optional[list] = None


def normalize_config(config: dict) -> dict:
    """설정 정규화 및 기본값 적용"""
    if "environment" not in config:
        config["environment"] = {}

    env = config["environment"]
    env.setdefault("type", "warehouse")
    env.setdefault("height", 5)
    env.setdefault("floor_friction", 0.7)
    env.setdefault("ambient_temperature", 25)
    env.setdefault("humidity", 50)

    if "robots" not in config:
        config["robots"] = []

    for robot in config.get("robots", []):
        robot.setdefault("type", "AGV")
        robot.setdefault("speed", 1.5 if robot.get("type") == "AGV" else 2.0)
        robot.setdefault("acceleration", 1.0)
        robot.setdefault("payload", 100)
        robot.setdefault("battery_capacity", 100)
        robot.setdefault("collision_radius", 0.5)
        robot.setdefault("sensor_range", 5.0)

    for obstacle in config.get("obstacles", []):
        obstacle.setdefault("type", "shelf")
        obstacle.setdefault("arrangement", "row")
        obstacle.setdefault("width", 1.0)
        obstacle.setdefault("length", 2.0)
        obstacle.setdefault("height", 2.0)
        obstacle.setdefault("spacing", 3.0)

    if "simulation" not in config:
        config["simulation"] = {}

    sim = config["simulation"]
    sim.setdefault("duration", 3600)
    sim.setdefault("time_step", 0.01)
    sim.setdefault("realtime_factor", 1.0)
    sim.setdefault("enable_physics", True)
    sim.setdefault("enable_collision", True)
    sim.setdefault("record_trajectory", True)

    return config


def validate_config(config: dict) -> Tuple[bool, list]:
    """설정 검증"""
    errors = []
    try:
        validated = validate_and_normalize(config)
        return True, []
    except Exception as e:
        if hasattr(e, 'errors'):
            for err in e.errors():
                loc = " → ".join(str(x) for x in err.get('loc', []))
                msg = err.get('msg', str(err))
                errors.append(f"{loc}: {msg}")
        else:
            errors.append(str(e))
        return False, errors


async def parse_environment_request(
    user_input: str,
    session_id: Optional[str] = None,
    documents: str = "",
    examples: str = "",
    force_provider: Optional[LLMProvider] = None,
    allow_external: bool = False  # v2.1: 보안 모드
) -> LLMResponse:
    """
    자연어 → Isaac Sim 파라미터 변환 (v2.1 보안 우선)

    기본: 로컬 LLM 사용 (데이터 외부 전송 없음)
    allow_external=True: 외부 API 허용 (빠른 응답)
    """
    session = session_store.get_or_create(session_id)
    session.add_user_message(user_input)

    # 의도 감지
    intent = session.detect_intent(user_input)

    # 수정 요청
    if intent == "modify" and session.get_current_config():
        return await modify_parameters(
            session.get_current_config(),
            user_input,
            session_id,
            force_provider,
            allow_external=allow_external
        )

    # 민감도 분류
    sensitivity = classify_sensitivity(user_input)

    # 프롬프트 구성
    prompt = build_env_prompt(
        user_input=user_input,
        documents=documents or "(관련 문서 없음)",
        examples=examples or "(유사 예시 없음)",
        conversation_context=session.get_context_summary()
    )

    try:
        # LLM 호출 (v2.1 보안 우선)
        raw_response, provider = await call_llm(
            prompt,
            temperature=0.3,
            sensitivity=sensitivity,
            force_provider=force_provider,
            allow_external=allow_external
        )

        logger.info(f"LLM Provider: {provider.value}, Sensitivity: {sensitivity.value}")

        # JSON 추출
        config, extract_error = extract_json(raw_response)

        if not config:
            return LLMResponse(
                success=False,
                data=None,
                raw_response=raw_response,
                provider=provider.value,
                sensitivity=sensitivity.value,
                error=f"JSON 추출 실패: {extract_error}"
            )

        # 정규화 및 검증
        config = normalize_config(config)
        is_valid, validation_errors = validate_config(config)

        # 세션 업데이트
        session.update_config(config)
        session.add_assistant_message(
            json.dumps(config, ensure_ascii=False)[:500],
            {"type": "config", "valid": is_valid, "provider": provider.value}
        )

        return LLMResponse(
            success=True,
            data=config,
            raw_response=raw_response,
            provider=provider.value,
            sensitivity=sensitivity.value,
            validation_errors=validation_errors if not is_valid else None
        )

    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        return LLMResponse(
            success=False,
            data=None,
            raw_response="",
            error=str(e)
        )


async def modify_parameters(
    current_config: dict,
    user_input: str,
    session_id: Optional[str] = None,
    force_provider: Optional[LLMProvider] = None,
    allow_external: bool = False  # v2.1: 보안 모드
) -> LLMResponse:
    """대화형 파라미터 수정 (v2.1 보안 우선)"""
    session = session_store.get_or_create(session_id)
    sensitivity = classify_sensitivity(user_input)

    prompt = build_modify_prompt(
        user_input=user_input,
        current_config=current_config,
        conversation_history=session.get_history_for_prompt(max_tokens=1000)
    )

    try:
        raw_response, provider = await call_llm(
            prompt,
            temperature=0.2,
            sensitivity=sensitivity,
            force_provider=force_provider,
            allow_external=allow_external
        )

        config, extract_error = extract_json(raw_response)

        if not config:
            return LLMResponse(
                success=False,
                data=None,
                raw_response=raw_response,
                provider=provider.value,
                error=f"JSON 추출 실패: {extract_error}"
            )

        # 부분 업데이트 병합
        if "environment" not in config and "robots" not in config:
            config = deep_merge(current_config, config)

        config = normalize_config(config)
        is_valid, validation_errors = validate_config(config)

        session.update_config(config)

        return LLMResponse(
            success=True,
            data=config,
            raw_response=raw_response,
            provider=provider.value,
            validation_errors=validation_errors if not is_valid else None
        )

    except Exception as e:
        return LLMResponse(
            success=False,
            data=None,
            raw_response="",
            error=str(e)
        )


async def analyze_simulation_result(
    sim_config: dict,
    sim_result: dict,
    documents: str = "",
    examples: str = "",
    force_provider: Optional[LLMProvider] = None
) -> LLMResponse:
    """시뮬레이션 결과 분석 (v2.0)"""
    prompt = build_analysis_prompt(
        sim_config=sim_config,
        sim_result=sim_result,
        documents=documents,
        examples=examples
    )

    # 시뮬레이션 결과는 내부 정보로 취급
    sensitivity = SensitivityLevel.INTERNAL

    try:
        raw_response, provider = await call_llm(
            prompt,
            temperature=0.5,
            max_tokens=3000,
            sensitivity=sensitivity,
            force_provider=force_provider
        )

        return LLMResponse(
            success=True,
            data=raw_response,
            raw_response=raw_response,
            provider=provider.value,
            sensitivity=sensitivity.value
        )

    except Exception as e:
        return LLMResponse(
            success=False,
            data=None,
            raw_response="",
            error=str(e)
        )


async def analyze_error(
    error_message: str,
    current_config: dict,
    force_provider: Optional[LLMProvider] = None
) -> LLMResponse:
    """에러 분석"""
    prompt = build_error_prompt(error_message, current_config)
    sensitivity = SensitivityLevel.INTERNAL

    try:
        raw_response, provider = await call_llm(
            prompt,
            temperature=0.3,
            sensitivity=sensitivity,
            force_provider=force_provider
        )

        config, _ = extract_json(raw_response)

        return LLMResponse(
            success=True,
            data={
                "analysis": raw_response,
                "suggested_config": config
            },
            raw_response=raw_response,
            provider=provider.value
        )

    except Exception as e:
        return LLMResponse(
            success=False,
            data=None,
            raw_response="",
            error=str(e)
        )


async def compare_simulations(
    config_a: dict, result_a: dict,
    config_b: dict, result_b: dict,
    force_provider: Optional[LLMProvider] = None
) -> LLMResponse:
    """시뮬레이션 비교 분석"""
    prompt = build_comparison_prompt(config_a, result_a, config_b, result_b)
    sensitivity = SensitivityLevel.INTERNAL

    try:
        raw_response, provider = await call_llm(
            prompt,
            temperature=0.4,
            max_tokens=3000,
            sensitivity=sensitivity,
            force_provider=force_provider
        )

        return LLMResponse(
            success=True,
            data=raw_response,
            raw_response=raw_response,
            provider=provider.value
        )

    except Exception as e:
        return LLMResponse(
            success=False,
            data=None,
            raw_response="",
            error=str(e)
        )


def deep_merge(base: dict, update: dict) -> dict:
    """딕셔너리 깊은 병합"""
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            for i, item in enumerate(value):
                if i < len(result[key]) and isinstance(item, dict):
                    result[key][i] = deep_merge(result[key][i], item)
                elif i < len(result[key]):
                    result[key][i] = item
                else:
                    result[key].append(item)
        else:
            result[key] = value

    return result


# 세션 관리
def get_session(session_id: str) -> Optional[ConversationManager]:
    return session_store.get(session_id)


def create_session() -> ConversationManager:
    return session_store.get_or_create()


def delete_session(session_id: str):
    session_store.delete(session_id)
