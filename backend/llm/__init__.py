"""
Forge LLM Module - 완전한 LLM 서비스
"""
from .service import (
    # 핵심 함수
    parse_environment_request,
    modify_parameters,
    analyze_simulation_result,
    analyze_error,
    compare_simulations,

    # 응답 타입
    LLMResponse,
    LLMError,

    # 유틸리티
    extract_json,
    normalize_config,
    validate_config,
    deep_merge,

    # 세션 관리
    get_session,
    create_session,
    delete_session
)

from .schemas import (
    IsaacSimParameters,
    Environment,
    Robot,
    Obstacle,
    SimulationConfig,
    TaskConfig,
    SimulationResult,
    validate_and_normalize,
    get_schema_json
)

from .conversation import (
    ConversationManager,
    MessageRole,
    session_store
)

from .prompts import (
    SYSTEM_PROMPT,
    build_env_prompt,
    build_modify_prompt,
    build_analysis_prompt
)

from .providers import (
    classify_sensitivity,
    classify_sensitivity_detailed,
    SensitivityResult,
    SensitivityLevel,
    SensitivityClassifier,
    LLMProvider
)

__all__ = [
    # 서비스
    "parse_environment_request",
    "modify_parameters",
    "analyze_simulation_result",
    "analyze_error",
    "compare_simulations",

    # 타입
    "LLMResponse",
    "LLMError",
    "IsaacSimParameters",

    # 스키마
    "validate_and_normalize",
    "get_schema_json",

    # 세션
    "get_session",
    "create_session",
    "delete_session",
    "ConversationManager",

    # 민감도 분류
    "classify_sensitivity",
    "classify_sensitivity_detailed",
    "SensitivityResult",
    "SensitivityLevel",
    "SensitivityClassifier",
    "LLMProvider",

    # 유틸
    "normalize_config",
    "validate_config"
]
