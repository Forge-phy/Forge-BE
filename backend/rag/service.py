"""
Forge RAG Service - 검색 증강 생성 (v2.0 - 민감도 분류 적용)
"""
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from .vector_store import search, search_similar_examples, add_feedback, add_example

import sys
sys.path.insert(0, '/Users/yangjeong-u/Projects/forge')
from llm.providers import (
    classify_sensitivity_detailed,
    call_llm,
    call_ollama,
    SensitivityLevel,
    LLMProvider,
    SensitivityResult,
    extract_json
)
from llm.service import normalize_config
from llm.prompts import SYSTEM_PROMPT, build_env_prompt, build_analysis_prompt


@dataclass
class RAGResponse:
    """RAG 응답 결과"""
    data: dict
    provider: str
    sensitivity: str
    input_sensitivity: dict
    context_sensitivity: dict
    reason: str


def format_documents(docs: List[Dict]) -> str:
    """검색된 문서 포맷팅"""
    if not docs:
        return "(관련 문서 없음)"

    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.get('content', '')[:500]
        formatted.append(f"[문서 {i}]\n{content}")
    return "\n\n".join(formatted)


def format_examples(examples: List[Dict]) -> str:
    """유사 예시 포맷팅"""
    if not examples:
        return "(유사 예시 없음)"

    formatted = []
    for i, ex in enumerate(examples, 1):
        content = ex.get('content', '')[:400]
        formatted.append(f"[예시 {i}]\n{content}")
    return "\n\n".join(formatted)


def _get_highest_sensitivity(
    input_result: SensitivityResult,
    context_result: SensitivityResult
) -> Tuple[SensitivityLevel, str]:
    """
    입력과 컨텍스트 중 더 높은 민감도 선택

    민감도 순서: CONFIDENTIAL > INTERNAL > PUBLIC
    """
    sensitivity_order = {
        SensitivityLevel.PUBLIC: 0,
        SensitivityLevel.INTERNAL: 1,
        SensitivityLevel.CONFIDENTIAL: 2
    }

    input_score = sensitivity_order[input_result.level]
    context_score = sensitivity_order[context_result.level]

    if context_score > input_score:
        return context_result.level, f"검색 결과에서 민감 정보 감지: {context_result.reason}"
    elif input_score > 0:
        return input_result.level, f"사용자 입력에서 민감 정보 감지: {input_result.reason}"
    else:
        return SensitivityLevel.PUBLIC, "일반 정보"


async def rag_parse_environment(user_input: str, allow_external: bool = False) -> RAGResponse:
    """
    RAG 기반 환경 파라미터 생성 (v2.1 - 보안 우선)

    기본: 로컬 LLM 사용 (데이터 외부 전송 없음)
    allow_external=True: 외부 API 허용 (빠른 응답)

    B안: 사용자 입력 + 검색 결과 모두 민감도 체크
    """
    # ========== 1. 관련 문서/예시 검색 ==========
    docs = search(user_input, "documents", n_results=3)
    examples = search_similar_examples(user_input, n_results=2)

    # 포맷팅
    docs_text = format_documents(docs)
    examples_text = format_examples(examples)

    # ========== 2. 민감도 분류 (B안: 입력 + 검색 결과 모두) ==========
    # 2-1. 사용자 입력 민감도
    input_sensitivity = classify_sensitivity_detailed(user_input)

    # 2-2. 검색 결과 민감도 (문서 + 예시 합쳐서)
    context_text = f"{docs_text}\n{examples_text}"
    context_sensitivity = classify_sensitivity_detailed(context_text)

    # 2-3. 더 높은 민감도 선택
    final_sensitivity, reason = _get_highest_sensitivity(
        input_sensitivity,
        context_sensitivity
    )

    # v2.1: 보안 우선 모드에서는 민감도와 무관하게 internal로 표시
    if not allow_external:
        reason = "보안 모드 (로컬 처리)"

    # ========== 3. 프롬프트 구성 ==========
    prompt = build_env_prompt(
        user_input=user_input,
        documents=docs_text,
        examples=examples_text,
        conversation_context="(새 대화)"
    )

    # ========== 4. LLM 호출 (v2.1 보안 우선) ==========
    response, provider = await call_llm(
        prompt,
        temperature=0.3,
        sensitivity=final_sensitivity,
        allow_external=allow_external
    )

    # ========== 5. JSON 추출 및 정규화 ==========
    config, error = extract_json(response)

    if config:
        config = normalize_config(config)

    return RAGResponse(
        data=config or {},
        provider=provider.value,
        sensitivity="internal" if not allow_external else final_sensitivity.value,
        input_sensitivity=input_sensitivity.to_dict(),
        context_sensitivity=context_sensitivity.to_dict(),
        reason=reason
    )


@dataclass
class RAGAnalysisResponse:
    """RAG 분석 응답 결과"""
    analysis: str
    provider: str
    sensitivity: str
    reason: str


async def rag_analyze_result(sim_config: dict, sim_result: dict) -> RAGAnalysisResponse:
    """
    RAG 기반 결과 분석 (v2.0 - 민감도 분류 적용)
    """
    # ========== 1. 관련 문서/예시 검색 ==========
    query = f"시뮬레이션 분석 {sim_result.get('bottleneck', '')} 충돌 최적화"
    docs = search(query, "documents", n_results=3)
    examples = search_similar_examples(query, n_results=2)

    docs_text = format_documents(docs)
    examples_text = format_examples(examples)

    # ========== 2. 민감도 분류 ==========
    # 시뮬 결과에는 민감 데이터가 포함될 수 있음
    input_text = f"{json.dumps(sim_config)} {json.dumps(sim_result)}"
    input_sensitivity = classify_sensitivity_detailed(input_text)

    context_text = f"{docs_text}\n{examples_text}"
    context_sensitivity = classify_sensitivity_detailed(context_text)

    final_sensitivity, reason = _get_highest_sensitivity(
        input_sensitivity,
        context_sensitivity
    )

    # ========== 3. 프롬프트 구성 ==========
    prompt = build_analysis_prompt(
        sim_config=sim_config,
        sim_result=sim_result,
        documents=docs_text,
        examples=examples_text
    )

    # ========== 4. LLM 호출 ==========
    response, provider = await call_llm(
        prompt,
        temperature=0.5,
        max_tokens=3000,
        sensitivity=final_sensitivity
    )

    return RAGAnalysisResponse(
        analysis=response,
        provider=provider.value,
        sensitivity=final_sensitivity.value,
        reason=reason
    )


def submit_feedback(user_input: str, llm_output: dict, is_positive: bool, comment: str = "") -> str:
    """
    사용자 피드백 제출 (지속학습의 핵심)

    - 긍정적 피드백: 좋은 예시로 저장 → 이후 유사 질문에 참조됨
    - 부정적 피드백: 피드백 DB에만 저장 (추후 분석용)
    """
    feedback_text = f"{'좋음' if is_positive else '나쁨'}: {comment}"
    return add_feedback(user_input, llm_output, feedback_text, is_positive)


def manually_add_good_example(user_input: str, correct_output: dict) -> str:
    """수동으로 좋은 예시 추가"""
    return add_example(user_input, correct_output, "good")
