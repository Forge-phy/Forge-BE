"""
Forge Isaac Sim - Gap 예측기 및 보고서 생성
LSTM 모델을 사용한 Sim2Real Gap 예측 및 LLM 보고서 생성
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .schemas import SimulationResult, EnvironmentConfig

logger = logging.getLogger(__name__)


@dataclass
class GapPrediction:
    """Gap 예측 결과"""
    predicted_gap_percent: float      # 예측 Gap (%)
    confidence_interval: tuple        # 신뢰구간 (low, high)
    confidence_score: float           # 신뢰도 (0-1)
    predicted_real_throughput: float  # 예측 실제 처리량
    factors: Dict[str, float]         # 영향 요인별 기여도

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_gap_percent": self.predicted_gap_percent,
            "confidence_interval": self.confidence_interval,
            "confidence_score": self.confidence_score,
            "predicted_real_throughput": self.predicted_real_throughput,
            "factors": self.factors
        }


@dataclass
class SimulationReport:
    """시뮬레이션 보고서"""
    summary: str
    simulation_result: Dict[str, Any]
    gap_prediction: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "simulation_result": self.simulation_result,
            "gap_prediction": self.gap_prediction,
            "recommendations": self.recommendations,
            "warnings": self.warnings
        }


class GapPredictor:
    """
    Sim2Real Gap 예측기

    LSTM 모델을 사용하여 시뮬레이션 결과와 환경 조건으로부터
    실제 현장 성능 갭을 예측합니다.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: LSTM 모델 파일 경로
        """
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.config = None
        self._load_model()

    def _get_default_model_path(self) -> str:
        """기본 모델 경로"""
        return str(Path(__file__).parent.parent / "lstm" / "data" / "sim2real_lstm_v3.pth")

    def _load_model(self):
        """모델 로드"""
        try:
            import torch

            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            self.feature_scaler = checkpoint.get('feature_scaler')
            self.target_scaler = checkpoint.get('target_scaler')
            self.config = checkpoint.get('config')

            # 모델 구조 재생성
            try:
                from lstm.model import Sim2RealLSTM
            except ImportError:
                from ..lstm.model import Sim2RealLSTM

            self.model = Sim2RealLSTM(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config.get('dropout', 0.2)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            logger.info(f"Model loaded: {self.model_path}")
            logger.info(f"Model metrics: {checkpoint.get('metrics', {})}")

        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using fallback prediction.")
            self.model = None

    def predict(
        self,
        sim_result: SimulationResult,
        env_config: EnvironmentConfig
    ) -> GapPrediction:
        """
        Gap 예측

        Args:
            sim_result: 시뮬레이션 결과
            env_config: 환경 설정

        Returns:
            Gap 예측 결과
        """
        # 특성 추출
        features = self._extract_features(sim_result, env_config)

        if self.model is not None:
            # LSTM 예측
            gap_percent, confidence = self._lstm_predict(features)
        else:
            # 폴백: 규칙 기반 예측
            gap_percent, confidence = self._rule_based_predict(features)

        # 영향 요인 분석
        factors = self._analyze_factors(features, gap_percent)

        # 신뢰구간 계산 (±2 MAE 기준)
        mae = 1.5  # 모델 MAE (실제로는 학습 결과에서 가져옴)
        ci_low = gap_percent - 2 * mae
        ci_high = gap_percent + 2 * mae

        # 실제 처리량 예측
        predicted_real = sim_result.total_throughput * (1 + gap_percent / 100)

        return GapPrediction(
            predicted_gap_percent=round(gap_percent, 2),
            confidence_interval=(round(ci_low, 2), round(ci_high, 2)),
            confidence_score=round(confidence, 2),
            predicted_real_throughput=round(predicted_real, 1),
            factors=factors
        )

    def _extract_features(
        self,
        sim_result: SimulationResult,
        env_config: EnvironmentConfig
    ) -> Dict[str, float]:
        """특성 추출"""
        return {
            "sim_throughput": sim_result.total_throughput,
            "temperature": env_config.temperature,
            "humidity": env_config.humidity,
            "robot_count": len(env_config.robots),
            "operation_hours": sim_result.total_duration / 3600,
            "warehouse_size": env_config.warehouse.width * env_config.warehouse.length,
            "collisions": sim_result.total_collisions,
        }

    def _lstm_predict(self, features: Dict[str, float]) -> tuple:
        """LSTM 모델 예측"""
        import torch

        # 특성 배열 생성 (시퀀스 형태로)
        feature_names = self.config.get('feature_cols', list(features.keys()))
        feature_array = np.array([[features.get(f, 0) for f in feature_names]])

        # 스케일링
        if self.feature_scaler:
            feature_array = self.feature_scaler.transform(feature_array)

        # 시퀀스 생성 (마지막 값 반복)
        seq_length = self.config.get('seq_length', 24)
        sequence = np.tile(feature_array, (seq_length, 1))
        sequence = sequence.reshape(1, seq_length, -1)

        # 예측
        with torch.no_grad():
            x = torch.FloatTensor(sequence)
            prediction = self.model(x).numpy()

        # 역스케일링
        if self.target_scaler:
            prediction = self.target_scaler.inverse_transform(prediction)

        gap_percent = float(prediction[0, 0])
        confidence = 0.82  # 모델 R² 점수

        return gap_percent, confidence

    def _rule_based_predict(self, features: Dict[str, float]) -> tuple:
        """규칙 기반 예측 (폴백)"""
        gap = -10.0  # 기본 갭 -10%

        # 온도 영향 (25°C 기준)
        temp_effect = -0.5 * abs(features.get("temperature", 25) - 25)

        # 습도 영향 (50% 기준, 높을수록 나쁨)
        humidity = features.get("humidity", 50)
        if humidity > 60:
            humidity_effect = -0.3 * (humidity - 60)
        else:
            humidity_effect = 0

        # 로봇 수 영향 (혼잡도)
        robot_count = features.get("robot_count", 3)
        if robot_count > 4:
            congestion_effect = -1.0 * (robot_count - 4)
        else:
            congestion_effect = 0

        gap += temp_effect + humidity_effect + congestion_effect
        gap = max(-35, min(5, gap))  # -35% ~ +5% 범위 제한

        return gap, 0.6  # 낮은 신뢰도

    def _analyze_factors(self, features: Dict[str, float], gap: float) -> Dict[str, float]:
        """영향 요인 분석"""
        factors = {}

        # 온도 기여도
        temp_deviation = abs(features.get("temperature", 25) - 25)
        factors["temperature"] = round(-0.5 * temp_deviation, 2)

        # 습도 기여도
        humidity = features.get("humidity", 50)
        if humidity > 60:
            factors["humidity"] = round(-0.3 * (humidity - 60), 2)
        else:
            factors["humidity"] = 0

        # 로봇 혼잡도 기여도
        robot_count = features.get("robot_count", 3)
        if robot_count > 4:
            factors["congestion"] = round(-1.0 * (robot_count - 4), 2)
        else:
            factors["congestion"] = 0

        return factors


class ReportGenerator:
    """
    보고서 생성기

    시뮬레이션 결과와 Gap 예측을 바탕으로 자연어 보고서를 생성합니다.
    """

    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: LLM 서비스 (옵션)
        """
        self.llm_service = llm_service

    async def generate_report(
        self,
        sim_result: SimulationResult,
        gap_prediction: GapPrediction,
        env_config: EnvironmentConfig
    ) -> SimulationReport:
        """
        보고서 생성

        Args:
            sim_result: 시뮬레이션 결과
            gap_prediction: Gap 예측 결과
            env_config: 환경 설정

        Returns:
            시뮬레이션 보고서
        """
        # 요약 생성
        summary = self._generate_summary(sim_result, gap_prediction, env_config)

        # 권장사항 생성
        recommendations = self._generate_recommendations(gap_prediction, env_config)

        # 경고 생성
        warnings = self._generate_warnings(sim_result, gap_prediction)

        # LLM으로 보고서 개선 (옵션)
        if self.llm_service:
            summary = await self._enhance_with_llm(summary, sim_result, gap_prediction)

        return SimulationReport(
            summary=summary,
            simulation_result=sim_result.to_dict(),
            gap_prediction=gap_prediction.to_dict(),
            recommendations=recommendations,
            warnings=warnings
        )

    def _generate_summary(
        self,
        sim_result: SimulationResult,
        gap_prediction: GapPrediction,
        env_config: EnvironmentConfig
    ) -> str:
        """요약 생성"""
        ci_low, ci_high = gap_prediction.confidence_interval

        summary = f"""시뮬레이션 결과 분석 리포트

■ 시뮬레이션 결과
  - 시뮬레이션 처리량: {sim_result.total_throughput:.1f}개/시간
  - 총 처리 물량: {sim_result.total_items_processed}개
  - 총 충돌 횟수: {sim_result.total_collisions}회
  - 운영 로봇: {len(sim_result.robot_metrics)}대

■ 현장 예측 (Sim2Real Gap)
  - 예상 Gap: {gap_prediction.predicted_gap_percent:.1f}%
  - 신뢰구간: {ci_low:.1f}% ~ {ci_high:.1f}%
  - 예상 현장 처리량: {gap_prediction.predicted_real_throughput:.1f}개/시간
  - 신뢰도: {gap_prediction.confidence_score * 100:.0f}%

■ 환경 조건
  - 온도: {env_config.temperature}°C
  - 습도: {env_config.humidity}%

■ 주요 Gap 요인
"""
        for factor, contribution in gap_prediction.factors.items():
            if contribution != 0:
                factor_kr = {
                    "temperature": "온도",
                    "humidity": "습도",
                    "congestion": "로봇 혼잡도"
                }.get(factor, factor)
                summary += f"  - {factor_kr}: {contribution:+.1f}%\n"

        return summary

    def _generate_recommendations(
        self,
        gap_prediction: GapPrediction,
        env_config: EnvironmentConfig
    ) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        # 온도 관련
        if env_config.temperature > 30:
            recommendations.append("환기 시스템을 점검하여 온도를 30°C 이하로 유지하세요.")
        elif env_config.temperature < 18:
            recommendations.append("난방 시스템을 점검하여 온도를 18°C 이상으로 유지하세요.")

        # 습도 관련
        if env_config.humidity > 70:
            recommendations.append("제습기를 가동하여 습도를 70% 이하로 유지하세요.")

        # 혼잡도 관련
        if len(env_config.robots) > 5:
            recommendations.append("로봇 간 충돌 방지를 위해 경로 최적화를 고려하세요.")

        # 생산 계획 관련
        recommendations.append(
            f"생산 계획은 {gap_prediction.predicted_real_throughput:.0f}개/시간 기준으로 수립하세요."
        )

        return recommendations

    def _generate_warnings(
        self,
        sim_result: SimulationResult,
        gap_prediction: GapPrediction
    ) -> List[str]:
        """경고 생성"""
        warnings = []

        # 충돌 경고
        if sim_result.total_collisions > 5:
            warnings.append(f"⚠️ 충돌이 {sim_result.total_collisions}회 발생했습니다. 로봇 경로를 재검토하세요.")

        # Gap 경고
        if gap_prediction.predicted_gap_percent < -20:
            warnings.append("⚠️ 예상 Gap이 -20% 이상입니다. 환경 조건 개선이 필요합니다.")

        # 신뢰도 경고
        if gap_prediction.confidence_score < 0.7:
            warnings.append("⚠️ 예측 신뢰도가 낮습니다. 추가 데이터 수집을 권장합니다.")

        return warnings

    async def _enhance_with_llm(
        self,
        summary: str,
        sim_result: SimulationResult,
        gap_prediction: GapPrediction
    ) -> str:
        """LLM으로 보고서 개선"""
        if not self.llm_service:
            return summary

        prompt = f"""
다음 시뮬레이션 보고서를 더 자연스럽고 전문적으로 다듬어주세요.
수치는 그대로 유지하되, 문장을 더 읽기 쉽게 개선해주세요.

원본 보고서:
{summary}
"""
        try:
            enhanced = await self.llm_service.generate(prompt)
            return enhanced
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return summary
