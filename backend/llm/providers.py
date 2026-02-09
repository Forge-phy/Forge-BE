"""
Forge LLM Providers - OpenAI + Ollama(Qwen) 하이브리드
v2.0: 민감도에 따라 LLM 분기
"""
import os
import json
import re
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum
import httpx

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI
from openai import AsyncOpenAI

from .prompts import SYSTEM_PROMPT


class LLMProvider(str, Enum):
    OPENAI = "openai"       # 일반 정보 - 빠르고 편리
    OLLAMA = "ollama"       # 민감 정보 - 로컬, 보안


class SensitivityLevel(str, Enum):
    PUBLIC = "public"           # 일반 정보 → OpenAI
    INTERNAL = "internal"       # 내부 정보 → Qwen 로컬
    CONFIDENTIAL = "confidential"  # 기밀 정보 → Qwen 로컬


@dataclass
class LLMConfig:
    """LLM 설정"""
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"  # 비용 효율적

    # Ollama (Qwen)
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "qwen2.5:7b"  # 또는 llama3

    # 기본 설정 - v2.1: 보안 우선 (기본 로컬)
    default_provider: LLMProvider = LLMProvider.OLLAMA
    security_first: bool = True  # True면 항상 로컬 우선
    timeout: float = 120.0


# 글로벌 설정
_config: Optional[LLMConfig] = None


def init_config(
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-4o-mini",
    ollama_model: str = "qwen2.5:7b",
    ollama_url: Optional[str] = None,
    default_provider: str = "ollama",  # v2.1: 기본 로컬
    security_first: bool = True  # v2.1: 보안 우선 모드
):
    """LLM 설정 초기화"""
    global _config

    # 환경 변수에서 API 키 가져오기
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    # 환경 변수에서 Ollama URL 가져오기
    ollama_host = ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # /api/generate 경로 추가
    if not ollama_host.endswith("/api/generate"):
        ollama_host = ollama_host.rstrip("/") + "/api/generate"

    # 환경 변수에서 보안 설정 가져오기 (SECURITY_FIRST=false로 비활성화 가능)
    env_security = os.getenv("SECURITY_FIRST", "true").lower()
    security_first = security_first and env_security != "false"

    _config = LLMConfig(
        openai_api_key=api_key,
        openai_model=openai_model,
        ollama_model=ollama_model,
        ollama_url=ollama_host,
        default_provider=LLMProvider(default_provider),
        security_first=security_first
    )

    return _config


def get_config() -> LLMConfig:
    """현재 설정 가져오기"""
    global _config
    if _config is None:
        # 환경 변수에서 Ollama URL 가져오기
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not ollama_host.endswith("/api/generate"):
            ollama_host = ollama_host.rstrip("/") + "/api/generate"

        # 환경 변수에서 보안 설정 가져오기
        env_security = os.getenv("SECURITY_FIRST", "true").lower()
        security_first = env_security != "false"

        _config = LLMConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            ollama_url=ollama_host,
            security_first=security_first
        )
    return _config


# ============================================================
# 민감도 분류기 v2.0 - 다단계 + 점수 기반
# ============================================================

@dataclass
class SensitivityResult:
    """민감도 분류 결과"""
    level: SensitivityLevel
    score: int
    reason: str
    is_simulation_context: bool
    detected_patterns: list
    detected_keywords: list

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "score": self.score,
            "reason": self.reason,
            "is_simulation_context": self.is_simulation_context,
            "detected_patterns": self.detected_patterns,
            "detected_keywords": self.detected_keywords
        }


class SensitivityClassifier:
    """
    다단계 + 점수 기반 민감도 분류기

    분류 파이프라인:
    1. 도메인 판단 (시뮬레이션 컨텍스트인지)
    2. 패턴 매칭 (실제 민감 데이터 정규식)
    3. 키워드 점수 (가중치 기반)
    4. 최종 판단 + 근거 반환
    """

    # 1. 시뮬레이션 도메인 키워드 (이 컨텍스트면 민감도 낮춤)
    SIMULATION_DOMAIN = [
        # 한글
        "시뮬레이션", "시뮬", "창고", "공장", "물류", "배치", "환경",
        "로봇", "agv", "amr", "컨베이어", "선반", "팔레트",
        # 영어
        "simulation", "warehouse", "factory", "logistics", "layout",
        "robot", "conveyor", "shelf", "pallet", "isaac", "sim"
    ]

    # 2. 실제 민감 데이터 패턴 (정규식)
    SENSITIVE_PATTERNS = {
        "주민번호": r'\d{6}[-\s]?\d{7}',
        "전화번호": r'01[016789][-\s]?\d{3,4}[-\s]?\d{4}',
        "카드번호": r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        "계좌번호": r'\d{3,4}[-\s]?\d{2,4}[-\s]?\d{4,6}',
        "이메일": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "API키_OpenAI": r'sk-[a-zA-Z0-9]{20,}',
        "API키_일반": r'[a-zA-Z0-9]{32,}',  # 32자 이상 연속 문자열
    }

    # 3. 카테고리별 민감 키워드 (가중치 포함)
    SENSITIVITY_RULES = {
        "CONFIDENTIAL": {
            "keywords": [
                # 인증 정보
                "비밀번호", "password", "passwd", "pwd",
                "api키", "api_key", "apikey", "secret_key",
                "토큰", "token", "access_token", "refresh_token",
                "인증키", "auth_key", "private_key",
            ],
            "weight": 100,
        },
        "INTERNAL_HIGH": {
            "keywords": [
                # 재무 정보
                "매출", "영업이익", "순이익", "revenue", "profit",
                "급여", "연봉", "salary", "wage", "compensation",
                "계약금액", "입찰가", "단가", "원가",
                # 인사 정보
                "인사평가", "성과급", "승진", "해고",
            ],
            "weight": 70,
        },
        "INTERNAL_MEDIUM": {
            "keywords": [
                # 고객/거래처 정보
                "고객명", "고객정보", "거래처", "협력사",
                "customer", "client", "vendor",
                # 내부 문서
                "대외비", "내부용", "미공개", "기밀",
                "confidential", "internal_only", "restricted",
            ],
            "weight": 50,
        },
        "INTERNAL_LOW": {
            "keywords": [
                # 일반 내부 정보
                "사내", "내부", "직원", "부서",
            ],
            "weight": 20,
        },
    }

    # 4. 시뮬레이션 컨텍스트에서 무시할 키워드
    # (이 키워드들은 시뮬레이션 맥락에서는 민감하지 않음)
    SIMULATION_SAFE_KEYWORDS = [
        "생산량", "production", "throughput",
        "설비", "장비", "equipment", "machine",
        "불량률", "defect", "error_rate",
        "가동률", "utilization", "efficiency",
    ]

    # 점수 임계값
    THRESHOLD_CONFIDENTIAL = 100
    THRESHOLD_INTERNAL = 40

    def __init__(self):
        # 패턴 컴파일 (성능 최적화)
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.SENSITIVE_PATTERNS.items()
        }

    def classify(self, text: str) -> SensitivityResult:
        """
        텍스트 민감도 분류

        Args:
            text: 분류할 텍스트

        Returns:
            SensitivityResult: 분류 결과 (레벨, 점수, 근거 포함)
        """
        text_lower = text.lower()

        detected_patterns = []
        detected_keywords = []
        score = 0
        reasons = []

        # ========== 1단계: 도메인 판단 ==========
        is_simulation = self._is_simulation_context(text_lower)

        # ========== 2단계: 패턴 매칭 ==========
        for pattern_name, compiled in self._compiled_patterns.items():
            if compiled.search(text):
                detected_patterns.append(pattern_name)
                score += 100  # 패턴 발견시 높은 점수
                reasons.append(f"민감 패턴 감지: {pattern_name}")

        # ========== 3단계: 키워드 점수 ==========
        for category, rules in self.SENSITIVITY_RULES.items():
            for keyword in rules["keywords"]:
                if keyword.lower() in text_lower:
                    # 시뮬레이션 컨텍스트에서 안전한 키워드는 스킵
                    if is_simulation and keyword.lower() in [k.lower() for k in self.SIMULATION_SAFE_KEYWORDS]:
                        continue

                    detected_keywords.append({
                        "keyword": keyword,
                        "category": category,
                        "weight": rules["weight"]
                    })
                    score += rules["weight"]
                    reasons.append(f"키워드 감지: {keyword} ({category})")

        # ========== 4단계: 최종 판단 ==========
        if score >= self.THRESHOLD_CONFIDENTIAL:
            level = SensitivityLevel.CONFIDENTIAL
            final_reason = "기밀 정보 감지"
        elif score >= self.THRESHOLD_INTERNAL:
            level = SensitivityLevel.INTERNAL
            final_reason = "내부 정보 감지"
        else:
            level = SensitivityLevel.PUBLIC
            final_reason = "일반 정보"

        # 상세 이유 추가
        if reasons:
            final_reason += f" ({', '.join(reasons[:3])})"  # 최대 3개만
        elif is_simulation:
            final_reason += " (시뮬레이션 컨텍스트)"

        return SensitivityResult(
            level=level,
            score=score,
            reason=final_reason,
            is_simulation_context=is_simulation,
            detected_patterns=detected_patterns,
            detected_keywords=detected_keywords
        )

    def _is_simulation_context(self, text_lower: str) -> bool:
        """시뮬레이션 관련 요청인지 판단"""
        match_count = sum(1 for kw in self.SIMULATION_DOMAIN if kw.lower() in text_lower)
        return match_count >= 2  # 2개 이상 매칭시 시뮬레이션 컨텍스트


# 글로벌 분류기 인스턴스
_classifier = SensitivityClassifier()


def classify_sensitivity(text: str) -> SensitivityLevel:
    """
    텍스트 민감도 분류 (하위 호환성 유지)

    Returns:
        SensitivityLevel: PUBLIC, INTERNAL, CONFIDENTIAL
    """
    result = _classifier.classify(text)
    return result.level


def classify_sensitivity_detailed(text: str) -> SensitivityResult:
    """
    텍스트 민감도 분류 (상세 결과 포함)

    Returns:
        SensitivityResult: 분류 결과 전체 (레벨, 점수, 근거 등)
    """
    return _classifier.classify(text)


def select_provider(
    sensitivity: SensitivityLevel,
    force_provider: Optional[LLMProvider] = None,
    allow_external: bool = False
) -> LLMProvider:
    """
    민감도에 따른 LLM 프로바이더 선택 (v2.1 보안 우선)

    기본 동작 (allow_external=False):
    - 모든 요청 → Ollama/Qwen (로컬) - 데이터 외부 전송 없음

    allow_external=True일 때:
    - OpenAI 사용 (빠름)
    """
    if force_provider:
        return force_provider

    config = get_config()

    # OpenAI API 키가 없으면 항상 Ollama
    if not config.openai_api_key:
        return LLMProvider.OLLAMA

    # allow_external=True면 OpenAI 사용 (사용자가 명시적으로 외부 API 허용)
    if allow_external:
        return LLMProvider.OPENAI

    # 기본: 로컬 사용 (security_first=True) 또는 민감 정보일 때
    return LLMProvider.OLLAMA


# ============================================================
# LLM 호출 함수
# ============================================================

async def call_openai(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    """OpenAI API 호출"""
    config = get_config()

    if not config.openai_api_key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다")

    client = AsyncOpenAI(api_key=config.openai_api_key)

    response = await client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


async def call_ollama(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    """Ollama API 호출 (Qwen 또는 Llama)"""
    config = get_config()

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        response = await client.post(
            config.ollama_url,
            json={
                "model": config.ollama_model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json().get("response", "")


async def call_llm(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    sensitivity: Optional[SensitivityLevel] = None,
    force_provider: Optional[LLMProvider] = None,
    allow_external: bool = False
) -> Tuple[str, LLMProvider]:
    """
    통합 LLM 호출 (v2.1 보안 우선)

    기본 동작:
    - 로컬 Qwen 사용 (데이터 외부 전송 없음)
    - Ollama 연결 실패 시 OpenAI로 폴백 (allow_external=True일 때만)

    Returns:
        (응답 텍스트, 사용된 프로바이더)
    """
    # 민감도 분류
    if sensitivity is None:
        sensitivity = classify_sensitivity(prompt)

    # 프로바이더 선택
    provider = select_provider(sensitivity, force_provider, allow_external)

    # LLM 호출
    if provider == LLMProvider.OPENAI:
        response = await call_openai(prompt, system, temperature, max_tokens)
    else:
        try:
            response = await call_ollama(prompt, system, temperature, max_tokens)
        except Exception as e:
            # Ollama 연결 실패 시 처리
            config = get_config()
            if config.openai_api_key and allow_external:
                # OpenAI로 폴백 (allow_external=True일 때만)
                print(f"⚠️ Ollama 연결 실패, OpenAI로 폴백: {e}")
                response = await call_openai(prompt, system, temperature, max_tokens)
                provider = LLMProvider.OPENAI
            else:
                # 외부 전송 불허용 시 에러 발생
                raise RuntimeError(
                    f"로컬 LLM(Ollama) 연결 실패: {e}. "
                    "보안 모드에서는 외부 API 사용이 제한됩니다. "
                    "Ollama 서비스를 시작하거나 allow_external=True로 설정하세요."
                )

    return response, provider


# ============================================================
# 헬퍼 함수
# ============================================================

def extract_json(text: str) -> Tuple[Optional[dict], str]:
    """텍스트에서 JSON 추출"""
    # 코드 블록 내 JSON
    code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_match:
        try:
            return json.loads(code_match.group(1)), ""
        except json.JSONDecodeError:
            pass

    # 중괄호 블록
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        try:
            depth = 0
            end_idx = 0
            for i, char in enumerate(json_match.group(1)):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

            json_str = json_match.group(1)[:end_idx]
            return json.loads(json_str), ""
        except json.JSONDecodeError:
            pass

    return None, "JSON을 찾을 수 없습니다"
