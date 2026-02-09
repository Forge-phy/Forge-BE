"""
Forge Isaac Sim - 자연어 파서 (LLM 브릿지)
자연어 입력을 Isaac Sim 파라미터로 변환
"""

import logging
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .schemas import (
    EnvironmentConfig, WarehouseConfig, RobotConfig, SimulationConfig,
    RobotType, Position, NLPRequest, NLPResponse
)

logger = logging.getLogger(__name__)


# 숫자 추출 패턴
NUMBER_PATTERN = r'(\d+(?:\.\d+)?)'

# 단위 변환
UNIT_CONVERSIONS = {
    "m": 1.0,
    "미터": 1.0,
    "meter": 1.0,
    "cm": 0.01,
    "센티": 0.01,
    "ft": 0.3048,
    "피트": 0.3048,
}


class NLPParser:
    """
    자연어 파서

    사용자의 자연어 입력을 분석하여 시뮬레이션 파라미터로 변환합니다.
    LLM과 연동하여 더 복잡한 요청도 처리할 수 있습니다.
    """

    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: LLM 서비스 (옵션, forge.llm.service.LLMService)
        """
        self.llm_service = llm_service

    async def parse(self, request: NLPRequest) -> NLPResponse:
        """
        자연어 요청 파싱

        Args:
            request: 자연어 요청

        Returns:
            파싱 결과 (환경/시뮬레이션 설정)
        """
        text = request.text.lower()
        logger.info(f"Parsing: {text}")

        try:
            # 1. 규칙 기반 파싱 시도
            env_config, sim_config, confidence = self._rule_based_parse(text)

            # 2. 신뢰도가 낮으면 LLM 사용
            if confidence < 0.7 and self.llm_service:
                env_config, sim_config, confidence = await self._llm_parse(text, request.context)

            return NLPResponse(
                success=True,
                environment_config=env_config,
                simulation_config=sim_config,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return NLPResponse(
                success=False,
                error_message=str(e),
                confidence=0.0
            )

    def _rule_based_parse(self, text: str) -> Tuple[EnvironmentConfig, SimulationConfig, float]:
        """규칙 기반 파싱"""
        env_config = EnvironmentConfig()
        sim_config = SimulationConfig()
        confidence = 0.5

        # 창고 크기 파싱
        warehouse_parsed = self._parse_warehouse_size(text)
        if warehouse_parsed:
            env_config.warehouse = warehouse_parsed
            confidence += 0.15

        # 로봇 수 파싱
        robots_parsed = self._parse_robots(text)
        if robots_parsed:
            env_config.robots = robots_parsed
            confidence += 0.15

        # 환경 조건 파싱
        temp, humidity = self._parse_environment_conditions(text)
        if temp is not None:
            env_config.temperature = temp
            confidence += 0.1
        if humidity is not None:
            env_config.humidity = humidity
            confidence += 0.1

        # 시뮬레이션 시간 파싱
        duration = self._parse_duration(text)
        if duration:
            sim_config.duration = duration
            confidence += 0.1

        return env_config, sim_config, min(confidence, 1.0)

    def _parse_warehouse_size(self, text: str) -> Optional[WarehouseConfig]:
        """창고 크기 파싱"""
        config = WarehouseConfig()

        # 패턴: "30m x 20m", "30미터 20미터", "가로 30 세로 20"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*[mx×]\s*(\d+(?:\.\d+)?)',  # 30x20, 30m x 20m
            r'(\d+(?:\.\d+)?)\s*미터?\s*[xX×]\s*(\d+(?:\.\d+)?)\s*미터?',
            r'가로\s*(\d+(?:\.\d+)?)\s*세로\s*(\d+(?:\.\d+)?)',
            r'너비\s*(\d+(?:\.\d+)?)\s*길이\s*(\d+(?:\.\d+)?)',
            r'width\s*(\d+(?:\.\d+)?)\s*length\s*(\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                config.width = float(match.group(1))
                config.length = float(match.group(2))
                logger.info(f"Parsed warehouse: {config.width}m x {config.length}m")
                return config

        # 단일 숫자 + "창고" 키워드
        if "창고" in text or "warehouse" in text:
            numbers = re.findall(NUMBER_PATTERN, text)
            if len(numbers) >= 2:
                config.width = float(numbers[0])
                config.length = float(numbers[1])
                return config

        return None

    def _parse_robots(self, text: str) -> Optional[list]:
        """로봇 설정 파싱"""
        robots = []

        # 로봇 수 파싱
        patterns = [
            r'로봇\s*(\d+)\s*대',
            r'AGV\s*(\d+)\s*대',
            r'(\d+)\s*대의?\s*로봇',
            r'(\d+)\s*robots?',
            r'robot\s*count[:\s]*(\d+)',
        ]

        robot_count = None
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                robot_count = int(match.group(1))
                break

        if robot_count is None:
            return None

        # 로봇 타입 파싱
        robot_type = RobotType.AGV  # 기본값
        if "amr" in text.lower():
            robot_type = RobotType.AMR
        elif "forklift" in text or "지게차" in text:
            robot_type = RobotType.FORKLIFT
        elif "arm" in text or "로봇팔" in text:
            robot_type = RobotType.ARM

        # 로봇 생성
        for i in range(robot_count):
            robots.append(RobotConfig(
                robot_id=f"{robot_type.value.upper()}_{i+1}",
                robot_type=robot_type,
                position=Position(x=2.0, y=2.0 + i * 2.0, z=0.0),
                speed=1.5
            ))

        logger.info(f"Parsed {len(robots)} robots ({robot_type.value})")
        return robots

    def _parse_environment_conditions(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """환경 조건 파싱"""
        temperature = None
        humidity = None

        # 온도 파싱
        temp_patterns = [
            r'온도\s*(\d+(?:\.\d+)?)\s*도?',
            r'(\d+(?:\.\d+)?)\s*°?[cC]',
            r'temperature[:\s]*(\d+(?:\.\d+)?)',
        ]
        for pattern in temp_patterns:
            match = re.search(pattern, text)
            if match:
                temperature = float(match.group(1))
                break

        # 습도 파싱
        humidity_patterns = [
            r'습도\s*(\d+(?:\.\d+)?)\s*%?',
            r'(\d+(?:\.\d+)?)\s*%\s*습도',
            r'humidity[:\s]*(\d+(?:\.\d+)?)',
        ]
        for pattern in humidity_patterns:
            match = re.search(pattern, text)
            if match:
                humidity = float(match.group(1))
                break

        return temperature, humidity

    def _parse_duration(self, text: str) -> Optional[float]:
        """시뮬레이션 시간 파싱"""
        patterns = [
            (r'(\d+)\s*시간', 3600),  # 시간 -> 초
            (r'(\d+)\s*분', 60),      # 분 -> 초
            (r'(\d+)\s*초', 1),       # 초
            (r'(\d+)\s*hours?', 3600),
            (r'(\d+)\s*minutes?', 60),
            (r'(\d+)\s*seconds?', 1),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier

        return None

    async def _llm_parse(self, text: str, context: Optional[Dict[str, Any]]) -> Tuple[EnvironmentConfig, SimulationConfig, float]:
        """LLM을 사용한 파싱"""
        if not self.llm_service:
            raise ValueError("LLM service not configured")

        # LLM 프롬프트 구성
        prompt = f"""
다음 자연어 요청을 분석하여 시뮬레이션 파라미터를 추출해주세요.

요청: "{text}"

다음 JSON 형식으로 응답해주세요:
{{
    "warehouse": {{
        "width": <숫자>,
        "length": <숫자>,
        "shelf_rows": <숫자>,
        "shelf_columns": <숫자>
    }},
    "robots": {{
        "count": <숫자>,
        "type": "agv" | "amr" | "forklift"
    }},
    "environment": {{
        "temperature": <숫자>,
        "humidity": <숫자>
    }},
    "simulation": {{
        "duration_hours": <숫자>
    }}
}}
"""

        # LLM 호출 (실제 구현 시)
        # response = await self.llm_service.generate(prompt)
        # parsed = json.loads(response)

        # PoC용 기본값 반환
        return EnvironmentConfig(), SimulationConfig(), 0.5


def parse_natural_language(text: str) -> Tuple[EnvironmentConfig, SimulationConfig]:
    """
    자연어를 시뮬레이션 설정으로 변환 (동기 헬퍼)

    Args:
        text: 자연어 입력

    Returns:
        (환경 설정, 시뮬레이션 설정)
    """
    parser = NLPParser()
    env_config, sim_config, _ = parser._rule_based_parse(text.lower())
    return env_config, sim_config


# 예시 사용법
EXAMPLE_QUERIES = [
    "30m x 20m 창고에 AGV 5대를 배치해줘",
    "온도 32도, 습도 70%인 환경에서 1시간 시뮬레이션",
    "로봇 3대로 물류 창고 시뮬레이션 해줘",
    "Create a 50x30 warehouse with 10 robots",
]
