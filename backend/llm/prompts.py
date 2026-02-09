"""
Forge LLM Prompts - 고급 프롬프트 엔지니어링
"""
from typing import Optional
import json

# ============================================================
# 시스템 프롬프트
# ============================================================

SYSTEM_PROMPT = """당신은 Forge의 AI 어시스턴트입니다. NVIDIA Isaac Sim 시뮬레이션 전문가로서 다음 역할을 수행합니다:

## 핵심 역할
1. **환경 설정**: 자연어 요청을 Isaac Sim 파라미터로 변환
2. **결과 분석**: 시뮬레이션 결과를 분석하고 인사이트 제공
3. **최적화 제안**: 성능 개선을 위한 구체적 권장사항 제시
4. **대화형 수정**: 기존 설정을 자연어로 수정

## 핵심 원칙
- 항상 유효한 Isaac Sim 파라미터만 생성
- 불확실한 값은 안전한 기본값 사용
- 물리적으로 불가능한 설정 거부
- 모든 숫자는 합리적인 범위 내

## 파라미터 범위
- 환경 크기: 1m ~ 10,000m
- 로봇 수: 1 ~ 1,000대
- 로봇 속도: 0.1 ~ 10.0 m/s (AGV 권장: 1.0-2.0, AMR 권장: 1.5-3.0)
- 시뮬레이션 시간: 1초 ~ 86,400초 (24시간)

## 응답 형식
- 환경 설정 요청: JSON만 출력
- 분석 요청: 한국어로 상세 설명
- 수정 요청: 수정된 JSON 출력
"""

# ============================================================
# 환경 설정 프롬프트 (고급)
# ============================================================

ENV_SETUP_PROMPT = """## 작업
사용자의 자연어 요청을 Isaac Sim 파라미터 JSON으로 변환하세요.
**추론이 필요한 경우 논리적으로 계산하여 값을 도출하세요.**

## 참고 문서
{documents}

## 유사 예시
{examples}

## 현재 대화 맥락
{conversation_context}

## 사용자 요청
{user_input}

## 추론 가이드라인 (중요!)
사용자가 간접적으로 크기/수량을 표현할 때 다음 기준으로 추론하세요:

### 창고 크기 추론
- "로봇 N대가 들어갈 수 있는 창고" → N대 × 100m² = 총 면적
  - 예: 100대 → 10,000m² → 약 100m × 100m
  - 예: 50대 → 5,000m² → 약 70m × 70m
- "대형 창고" → 80m × 60m 이상
- "중형 창고" → 40m × 30m
- "소형 창고" → 20m × 15m

### 로봇 밀도 추론
- 최적 밀도: 100m² 당 1대
- "여유있게" → 150m² 당 1대
- "빽빽하게" → 50m² 당 1대

### 선반 배치 추론
- "N열 선반" → obstacles의 count를 N으로 설정
- 선반 간격: 로봇 폭(약 1m)의 최소 2배 = 2m 이상

### 환경 조건 추론
- "더운 환경" → temperature: 35
- "추운 환경" → temperature: 10
- "습한 환경" → humidity: 80
- "건조한 환경" → humidity: 30

## 출력 형식
반드시 아래 JSON 구조를 따르세요. 누락된 값은 기본값을 사용하세요.

```json
{{
    "environment": {{
        "type": "warehouse" | "factory" | "custom",
        "width": <미터, 1-10000>,
        "length": <미터, 1-10000>,
        "height": <미터, 2-100, 기본 5>,
        "floor_friction": <0.1-1.0, 기본 0.7>,
        "ambient_temperature": <-20~60, 기본 25>,
        "humidity": <0-100, 기본 50>
    }},
    "robots": [
        {{
            "type": "AGV" | "AMR" | "arm" | "humanoid",
            "count": <1-1000>,
            "speed": <0.1-10.0 m/s, AGV 기본 1.5, AMR 기본 2.0>,
            "acceleration": <0.1-5.0, 기본 1.0>,
            "payload": <0-10000 kg, 기본 100>,
            "battery_capacity": <10-1000 Wh, 기본 100>,
            "collision_radius": <0.1-5.0 m, 기본 0.5>,
            "sensor_range": <0.5-50.0 m, 기본 5.0>
        }}
    ],
    "obstacles": [
        {{
            "type": "shelf" | "wall" | "conveyor" | "workstation" | "pillar",
            "count": <0-10000>,
            "arrangement": "grid" | "row" | "random" | "custom",
            "width": <0.1-100 m, 기본 1.0>,
            "length": <0.1-100 m, 기본 2.0>,
            "height": <0.1-50 m, 기본 2.0>,
            "spacing": <0.5-50 m, 기본 3.0>
        }}
    ],
    "simulation": {{
        "duration": <1-86400 초, 기본 3600>,
        "time_step": <0.001-1.0, 기본 0.01>,
        "realtime_factor": <0.1-100.0, 기본 1.0>,
        "enable_physics": <true/false, 기본 true>,
        "enable_collision": <true/false, 기본 true>,
        "record_trajectory": <true/false, 기본 true>
    }},
    "task": {{
        "type": "pickup_delivery" | "patrol" | "assembly" | "custom",
        "pickup_locations": <1-1000, 기본 5>,
        "delivery_locations": <1-1000, 기본 5>,
        "task_generation_rate": <0.001-10.0, 기본 0.1>
    }}
}}
```

## 중요 규칙
1. JSON만 출력하세요. 설명은 포함하지 마세요.
2. 사용자가 명시하지 않은 값은 합리적인 기본값을 사용하세요.
3. 로봇 밀도 권장: 100m² 당 1대 이하
4. 선반 간격은 로봇 폭의 최소 1.5배 이상
5. 물리적으로 불가능한 설정은 거부하세요.

JSON 출력:"""

# ============================================================
# 파라미터 수정 프롬프트 (대화형)
# ============================================================

PARAM_MODIFY_PROMPT = """## 작업
기존 설정을 사용자 요청에 따라 수정하세요.

## 현재 설정
```json
{current_config}
```

## 대화 히스토리
{conversation_history}

## 사용자 수정 요청
{user_input}

## 수정 규칙
1. 요청된 부분만 수정하고 나머지는 유지
2. "더", "추가" → 값 증가
3. "빼", "줄여" → 값 감소
4. "삭제", "제거" → 해당 항목 제거
5. 구체적 숫자가 있으면 그 값으로 설정
6. 상대적 표현은 합리적으로 해석 (예: "조금 올려" → 20% 증가)

## 수정 해석
- "속도 올려" → speed를 20-30% 증가
- "로봇 2대 더" → count에 2 추가
- "빠르게" → speed 증가, time_step 감소
- "정밀하게" → time_step 감소, sensor_range 증가

수정된 JSON만 출력하세요:"""

# ============================================================
# 결과 분석 프롬프트
# ============================================================

RESULT_ANALYSIS_PROMPT = """## 작업
시뮬레이션 결과를 분석하고 인사이트를 제공하세요.

## 참고 문서
{documents}

## 유사 분석 예시
{examples}

## 시뮬레이션 설정
```json
{sim_config}
```

## 시뮬레이션 결과
```json
{sim_result}
```

## 분석 요구사항
다음 항목을 포함하여 **한국어**로 분석하세요:

### 1. 핵심 지표 요약
- 처리량 (throughput)
- 효율성 (efficiency)
- 가동률 (utilization)
- 충돌 횟수 및 심각도

### 2. 문제점 식별
- 병목 지점 분석
- 충돌 패턴 분석
- 비효율 구간 식별

### 3. 원인 분석
- 각 문제의 근본 원인
- 설정과 결과의 상관관계

### 4. 개선 권장사항
구체적인 파라미터 변경을 포함하여 제안:
- 즉시 적용 가능한 변경
- 중기 개선 방안
- 장기 최적화 전략

### 5. Sim2Real 갭 예측
- 현장 적용 시 예상 성능 차이
- 주의해야 할 현실 요소

분석 결과:"""

# ============================================================
# 에러 분석 프롬프트
# ============================================================

ERROR_ANALYSIS_PROMPT = """## 작업
시뮬레이션 오류를 분석하고 해결책을 제시하세요.

## 오류 정보
```
{error_message}
```

## 현재 설정
```json
{current_config}
```

## 분석 요구사항
1. 오류 원인 파악
2. 설정 중 문제가 되는 부분 식별
3. 구체적인 수정 방안 제시
4. 수정된 설정 JSON 제공

한국어로 분석하세요:"""

# ============================================================
# 비교 분석 프롬프트
# ============================================================

COMPARISON_PROMPT = """## 작업
두 시뮬레이션 결과를 비교 분석하세요.

## 설정 A
```json
{config_a}
```

## 결과 A
```json
{result_a}
```

## 설정 B
```json
{config_b}
```

## 결과 B
```json
{result_b}
```

## 분석 요구사항
1. 주요 지표 비교 (테이블 형식)
2. 어떤 설정이 더 나은지 판단
3. 차이의 원인 분석
4. 최적 설정 권장

한국어로 분석하세요:"""

# ============================================================
# 프롬프트 빌더 함수
# ============================================================

def build_env_prompt(
    user_input: str,
    documents: str = "(관련 문서 없음)",
    examples: str = "(유사 예시 없음)",
    conversation_context: str = "(새 대화)"
) -> str:
    """환경 설정 프롬프트 생성"""
    return ENV_SETUP_PROMPT.format(
        user_input=user_input,
        documents=documents,
        examples=examples,
        conversation_context=conversation_context
    )


def build_modify_prompt(
    user_input: str,
    current_config: dict,
    conversation_history: str = ""
) -> str:
    """수정 프롬프트 생성"""
    return PARAM_MODIFY_PROMPT.format(
        user_input=user_input,
        current_config=json.dumps(current_config, indent=2, ensure_ascii=False),
        conversation_history=conversation_history or "(이전 대화 없음)"
    )


def build_analysis_prompt(
    sim_config: dict,
    sim_result: dict,
    documents: str = "(관련 문서 없음)",
    examples: str = "(유사 분석 없음)"
) -> str:
    """분석 프롬프트 생성"""
    return RESULT_ANALYSIS_PROMPT.format(
        sim_config=json.dumps(sim_config, indent=2, ensure_ascii=False),
        sim_result=json.dumps(sim_result, indent=2, ensure_ascii=False),
        documents=documents,
        examples=examples
    )


def build_error_prompt(error_message: str, current_config: dict) -> str:
    """에러 분석 프롬프트 생성"""
    return ERROR_ANALYSIS_PROMPT.format(
        error_message=error_message,
        current_config=json.dumps(current_config, indent=2, ensure_ascii=False)
    )


def build_comparison_prompt(
    config_a: dict, result_a: dict,
    config_b: dict, result_b: dict
) -> str:
    """비교 분석 프롬프트 생성"""
    return COMPARISON_PROMPT.format(
        config_a=json.dumps(config_a, indent=2, ensure_ascii=False),
        result_a=json.dumps(result_a, indent=2, ensure_ascii=False),
        config_b=json.dumps(config_b, indent=2, ensure_ascii=False),
        result_b=json.dumps(result_b, indent=2, ensure_ascii=False)
    )
