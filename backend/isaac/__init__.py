"""
Forge Isaac Sim Integration Module

Isaac Sim과 연동하여 시뮬레이션을 실행하고,
LSTM으로 Sim2Real Gap을 예측하며,
LLM으로 보고서를 생성합니다.

Usage:
    from forge.isaac import IsaacSimPipeline

    pipeline = IsaacSimPipeline()

    # 자연어로 시뮬레이션 실행
    result = await pipeline.run_from_natural_language(
        "30m x 20m 창고에 AGV 5대, 온도 32도, 습도 70% 환경에서 1시간 시뮬레이션"
    )

    print(result.report.summary)
"""

from .schemas import (
    EnvironmentConfig,
    WarehouseConfig,
    RobotConfig,
    SimulationConfig,
    SimulationResult,
    RobotType,
    Position,
    Rotation,
)

from .client import (
    IsaacSimClient,
    IsaacSimConfig,
    get_client,
    initialize_isaac_sim,
)

from .environment import (
    EnvironmentBuilder,
    create_default_warehouse_config,
)

from .simulation import (
    SimulationRunner,
    run_quick_simulation,
)

from .nlp_parser import (
    NLPParser,
    parse_natural_language,
)

from .predictor import (
    GapPredictor,
    GapPrediction,
    ReportGenerator,
    SimulationReport,
)

__all__ = [
    # Schemas
    "EnvironmentConfig",
    "WarehouseConfig",
    "RobotConfig",
    "SimulationConfig",
    "SimulationResult",
    "RobotType",
    "Position",
    "Rotation",
    # Client
    "IsaacSimClient",
    "IsaacSimConfig",
    "get_client",
    "initialize_isaac_sim",
    # Environment
    "EnvironmentBuilder",
    "create_default_warehouse_config",
    # Simulation
    "SimulationRunner",
    "run_quick_simulation",
    # NLP
    "NLPParser",
    "parse_natural_language",
    # Prediction
    "GapPredictor",
    "GapPrediction",
    "ReportGenerator",
    "SimulationReport",
    # Pipeline
    "IsaacSimPipeline",
]


# =============================================================================
# 통합 파이프라인
# =============================================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    success: bool
    environment_config: Optional[EnvironmentConfig] = None
    simulation_config: Optional[SimulationConfig] = None
    simulation_result: Optional[SimulationResult] = None
    gap_prediction: Optional[GapPrediction] = None
    report: Optional[SimulationReport] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "environment_config": self.environment_config.to_dict() if self.environment_config else None,
            "simulation_config": self.simulation_config.to_dict() if self.simulation_config else None,
            "simulation_result": self.simulation_result.to_dict() if self.simulation_result else None,
            "gap_prediction": self.gap_prediction.to_dict() if self.gap_prediction else None,
            "report": self.report.to_dict() if self.report else None,
            "error_message": self.error_message,
        }


class IsaacSimPipeline:
    """
    Forge Isaac Sim 통합 파이프라인

    전체 워크플로우를 통합합니다:
    1. 자연어 파싱 (LLM)
    2. 환경 생성
    3. 시뮬레이션 실행 (Isaac Sim)
    4. Gap 예측 (LSTM)
    5. 보고서 생성 (LLM)
    """

    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: LLM 서비스 인스턴스 (옵션)
        """
        self.llm_service = llm_service
        self.nlp_parser = NLPParser(llm_service)
        self.simulation_runner = SimulationRunner()
        self.gap_predictor = GapPredictor()
        self.report_generator = ReportGenerator(llm_service)

    async def run_from_natural_language(self, text: str) -> PipelineResult:
        """
        자연어 입력으로 전체 파이프라인 실행

        Args:
            text: 자연어 시뮬레이션 요청

        Returns:
            파이프라인 실행 결과
        """
        logger.info(f"Pipeline started: {text[:50]}...")

        try:
            # 1. 자연어 파싱
            from .schemas import NLPRequest
            parse_result = await self.nlp_parser.parse(NLPRequest(text=text))

            if not parse_result.success:
                return PipelineResult(
                    success=False,
                    error_message=f"파싱 실패: {parse_result.error_message}"
                )

            env_config = parse_result.environment_config
            sim_config = parse_result.simulation_config

            # 2. 시뮬레이션 실행
            sim_result = await self.simulation_runner.run_simulation(env_config, sim_config)

            if not sim_result.success:
                return PipelineResult(
                    success=False,
                    environment_config=env_config,
                    simulation_config=sim_config,
                    error_message=f"시뮬레이션 실패: {sim_result.error_message}"
                )

            # 3. Gap 예측
            gap_prediction = self.gap_predictor.predict(sim_result, env_config)

            # 4. 보고서 생성
            report = await self.report_generator.generate_report(
                sim_result, gap_prediction, env_config
            )

            logger.info("Pipeline completed successfully")

            return PipelineResult(
                success=True,
                environment_config=env_config,
                simulation_config=sim_config,
                simulation_result=sim_result,
                gap_prediction=gap_prediction,
                report=report
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(
                success=False,
                error_message=str(e)
            )

    async def run_from_config(
        self,
        env_config: EnvironmentConfig,
        sim_config: Optional[SimulationConfig] = None
    ) -> PipelineResult:
        """
        설정 객체로 파이프라인 실행

        Args:
            env_config: 환경 설정
            sim_config: 시뮬레이션 설정 (옵션)

        Returns:
            파이프라인 실행 결과
        """
        sim_config = sim_config or SimulationConfig()

        try:
            # 시뮬레이션 실행
            sim_result = await self.simulation_runner.run_simulation(env_config, sim_config)

            if not sim_result.success:
                return PipelineResult(
                    success=False,
                    environment_config=env_config,
                    simulation_config=sim_config,
                    error_message=f"시뮬레이션 실패: {sim_result.error_message}"
                )

            # Gap 예측
            gap_prediction = self.gap_predictor.predict(sim_result, env_config)

            # 보고서 생성
            report = await self.report_generator.generate_report(
                sim_result, gap_prediction, env_config
            )

            return PipelineResult(
                success=True,
                environment_config=env_config,
                simulation_config=sim_config,
                simulation_result=sim_result,
                gap_prediction=gap_prediction,
                report=report
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(
                success=False,
                error_message=str(e)
            )
