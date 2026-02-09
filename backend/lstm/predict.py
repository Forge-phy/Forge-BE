"""
Forge LSTM Prediction - 예측 서비스 (API 연동용)
v2.0: sim2real_lstm_v3.pth 지원
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime

from .model import LSTMConfig, create_model


@dataclass
class PredictionResult:
    """예측 결과"""
    sim_throughput: float
    predicted_real_throughput: float
    gap_ratio: float
    gap_percent: float
    confidence: float
    confidence_interval: tuple  # (lower, upper)
    factors: List[str]  # 주요 영향 요인


# ============================================================
# v3 모델용 LSTM 클래스
# ============================================================

class Sim2RealLSTMv3(nn.Module):
    """v3 모델 구조 (10개 입력 특성)"""
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class Sim2RealPredictor:
    """
    Sim2Real 갭 예측기

    Usage:
        predictor = Sim2RealPredictor()
        predictor.load_model("data/sim2real_lstm_v3.pth")

        result = predictor.predict(
            sim_throughput=100,
            temperature=32,
            humidity=75,
            robot_count=5,
            operation_hours=6,
            warehouse_size=600
        )
    """

    # v3 모델 특성 (10개)
    FEATURE_NAMES_V3 = [
        'sim_throughput', 'temperature', 'humidity', 'robot_count',
        'operation_hours', 'days_since_maintenance', 'hour',
        'day_of_week', 'is_weekend', 'day_of_year'
    ]

    # 기존 모델 특성 (6개)
    FEATURE_NAMES = [
        'sim_throughput', 'temperature', 'humidity',
        'robot_count', 'operation_hours', 'warehouse_size'
    ]

    # 영향 요인 분석용 기준값
    BASELINE = {
        'temperature': 25.0,
        'humidity': 50.0,
        'robot_count': 3,
        'operation_hours': 4.0
    }

    def __init__(self, model_type: str = "lstm"):
        self.model_type = model_type
        self.model = None
        self.config = None
        self.norm_params = None
        self.feature_scaler = None
        self.target_scaler = None
        self.is_v3 = False  # v3 모델 여부
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        norm_path: Optional[str] = None
    ):
        """모델 로드 (v3 및 기존 모델 모두 지원)"""
        model_path = Path(model_path)

        # v3 모델 체크 (.pth 확장자 + v3 이름)
        if 'v3' in model_path.stem or model_path.suffix == '.pth':
            self._load_v3_model(model_path)
        else:
            self._load_legacy_model(model_path, config_path, norm_path)

        print(f"모델 로드 완료: {model_path} ({'v3' if self.is_v3 else 'legacy'})")

    def _load_v3_model(self, model_path: Path):
        """v3 모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.config = checkpoint.get('config', {})
        self.feature_scaler = checkpoint.get('feature_scaler')
        self.target_scaler = checkpoint.get('target_scaler')
        self.is_v3 = True

        # v3 모델 생성
        self.model = Sim2RealLSTMv3(
            input_size=self.config.get('input_size', 10),
            hidden_size=self.config.get('hidden_size', 64),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.2)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _load_legacy_model(self, model_path: Path, config_path: Optional[str], norm_path: Optional[str]):
        """기존 모델 로드"""
        self.is_v3 = False

        # 설정 로드
        if config_path is None:
            config_path = model_path.parent / f"{model_path.stem.replace('_best', '_config')}.json"

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = LSTMConfig(**{k: v for k, v in config_dict.items()
                                        if k in LSTMConfig.__dataclass_fields__})
        else:
            self.config = LSTMConfig()

        # 정규화 파라미터 로드
        if norm_path is None:
            norm_path = model_path.parent / f"{model_path.stem.replace('_best', '_norm')}.json"

        if Path(norm_path).exists():
            with open(norm_path, 'r') as f:
                self.norm_params = json.load(f)

        # 모델 생성 및 가중치 로드
        self.model = create_model(self.model_type, self.config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """입력 정규화"""
        if self.is_v3 and self.feature_scaler is not None:
            return self.feature_scaler.transform(features)

        if self.norm_params is None:
            return features

        mean = np.array(self.norm_params['mean'])
        std = np.array(self.norm_params['std'])
        return (features - mean) / std

    def _denormalize_output(self, output: np.ndarray) -> np.ndarray:
        """출력 역정규화 (v3용)"""
        if self.is_v3 and self.target_scaler is not None:
            return self.target_scaler.inverse_transform(output.reshape(-1, 1)).flatten()
        return output

    def _analyze_factors(
        self,
        temperature: float,
        humidity: float,
        robot_count: int,
        operation_hours: float
    ) -> List[str]:
        """주요 영향 요인 분석"""
        factors = []

        # 온도 영향
        if temperature > self.BASELINE['temperature'] + 5:
            factors.append(f"고온 ({temperature:.1f}°C)")
        elif temperature < self.BASELINE['temperature'] - 5:
            factors.append(f"저온 ({temperature:.1f}°C)")

        # 습도 영향
        if humidity > self.BASELINE['humidity'] + 20:
            factors.append(f"고습도 ({humidity:.1f}%)")
        elif humidity < self.BASELINE['humidity'] - 15:
            factors.append(f"저습도 ({humidity:.1f}%)")

        # 혼잡도 영향
        if robot_count > self.BASELINE['robot_count'] + 3:
            factors.append(f"로봇 혼잡 ({robot_count}대)")

        # 피로도 영향
        if operation_hours > self.BASELINE['operation_hours'] + 4:
            factors.append(f"장시간 가동 ({operation_hours:.1f}시간)")

        if not factors:
            factors.append("정상 조건")

        return factors

    def predict(
        self,
        sim_throughput: float,
        temperature: float,
        humidity: float,
        robot_count: int,
        operation_hours: float,
        warehouse_size: float = 500.0,
        days_since_maintenance: int = None,
        return_dict: bool = False
    ) -> Union[PredictionResult, Dict]:
        """
        단일 예측

        Args:
            sim_throughput: 시뮬레이션 처리량 (개/시간)
            temperature: 현장 온도 (°C)
            humidity: 현장 습도 (%)
            robot_count: 로봇 수
            operation_hours: 연속 가동 시간
            warehouse_size: 창고 크기 (m²) - 기존 모델용
            days_since_maintenance: 정비 후 경과일 - v3 모델용 (자동 설정)
            return_dict: True면 dict 반환

        Returns:
            PredictionResult 또는 dict
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        # v3 모델용 특성 생성
        if self.is_v3:
            now = datetime.now()
            if days_since_maintenance is None:
                days_since_maintenance = 15  # 기본값: 2주

            features = np.array([[
                sim_throughput,
                temperature,
                humidity,
                robot_count,
                operation_hours,
                days_since_maintenance,
                now.hour,
                now.weekday(),
                1 if now.weekday() >= 5 else 0,
                now.timetuple().tm_yday
            ]], dtype=np.float32)
        else:
            # 기존 모델용 특성 (6개)
            features = np.array([[
                sim_throughput, temperature, humidity,
                robot_count, operation_hours, warehouse_size
            ]], dtype=np.float32)

        # 정규화
        features_norm = self._normalize(features)

        # 예측
        with torch.no_grad():
            x = torch.tensor(features_norm, dtype=torch.float32).to(self.device)
            raw_output = self.model(x).cpu().numpy()

            if self.is_v3:
                # v3: gap_percent를 직접 예측 (역정규화 필요)
                gap_percent_pred = float(self._denormalize_output(raw_output)[0])
                gap_ratio = gap_percent_pred / 100.0
                predicted_real = float(sim_throughput * (1 + gap_ratio))
                gap_percent = gap_percent_pred
            else:
                # 기존: gap_ratio 직접 출력
                gap_ratio = float(raw_output[0, 0])
                predicted_real = float(sim_throughput * (1 + gap_ratio))
                gap_percent = float(gap_ratio * 100)

        # 신뢰도 (갭이 작을수록 높은 신뢰도)
        confidence = max(0.5, min(0.99, 1 - abs(gap_ratio) * 2))

        # 신뢰구간 (±3% 가정)
        margin = abs(predicted_real * 0.03)
        confidence_interval = (
            round(predicted_real - margin, 1),
            round(predicted_real + margin, 1)
        )

        # 영향 요인 분석
        factors = self._analyze_factors(temperature, humidity, robot_count, operation_hours)

        result = PredictionResult(
            sim_throughput=sim_throughput,
            predicted_real_throughput=round(predicted_real, 2),
            gap_ratio=round(gap_ratio, 4),
            gap_percent=round(gap_percent, 2),
            confidence=round(confidence, 2),
            confidence_interval=confidence_interval,
            factors=factors
        )

        if return_dict:
            return {
                'sim_throughput': result.sim_throughput,
                'predicted_real_throughput': result.predicted_real_throughput,
                'gap_ratio': result.gap_ratio,
                'gap_percent': result.gap_percent,
                'confidence': result.confidence,
                'confidence_interval': list(result.confidence_interval),
                'factors': result.factors
            }

        return result

    def predict_batch(
        self,
        data: List[Dict]
    ) -> List[Dict]:
        """
        배치 예측

        Args:
            data: [{'sim_throughput': ..., 'temperature': ..., ...}, ...]

        Returns:
            예측 결과 리스트
        """
        results = []
        for item in data:
            result = self.predict(**item, return_dict=True)
            results.append(result)
        return results


# ============================================================
# 글로벌 예측기 인스턴스 (API 연동용)
# ============================================================

_predictor: Optional[Sim2RealPredictor] = None


def get_predictor() -> Sim2RealPredictor:
    """글로벌 예측기 가져오기 (v3 모델 우선)"""
    global _predictor
    if _predictor is None:
        _predictor = Sim2RealPredictor(model_type="lstm")

        # v3 모델 경로 (우선)
        v3_model_path = Path(__file__).parent / "data" / "sim2real_lstm_v3.pth"
        # 기존 모델 경로 (fallback)
        legacy_model_path = Path(__file__).parent / "checkpoints" / "mlp_sim2real_best.pt"

        if v3_model_path.exists():
            _predictor.load_model(str(v3_model_path))
        elif legacy_model_path.exists():
            _predictor.load_model(str(legacy_model_path))
        else:
            print(f"경고: 모델 파일 없음 - {v3_model_path} 또는 {legacy_model_path}")

    return _predictor


def predict_gap(
    sim_throughput: float,
    temperature: float = 25.0,
    humidity: float = 50.0,
    robot_count: int = 3,
    operation_hours: float = 4.0,
    warehouse_size: float = 500.0
) -> Dict:
    """
    Sim2Real 갭 예측 (간편 함수)

    Returns:
        {
            'sim_throughput': 100,
            'predicted_real_throughput': 85,
            'gap_percent': -15,
            'confidence': 0.89,
            'confidence_interval': [83, 87],
            'factors': ['고온 (32°C)', '고습도 (75%)']
        }
    """
    predictor = get_predictor()
    return predictor.predict(
        sim_throughput=sim_throughput,
        temperature=temperature,
        humidity=humidity,
        robot_count=robot_count,
        operation_hours=operation_hours,
        warehouse_size=warehouse_size,
        return_dict=True
    )


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Forge LSTM Prediction Test (v3)")
    print("=" * 60)

    # v3 모델 경로 (우선)
    v3_model_path = Path(__file__).parent / "data" / "sim2real_lstm_v3.pth"
    legacy_model_path = Path(__file__).parent / "checkpoints" / "mlp_sim2real_best.pt"

    if v3_model_path.exists():
        model_path = v3_model_path
    elif legacy_model_path.exists():
        model_path = legacy_model_path
    else:
        print(f"\n모델 파일이 없습니다")
        print("먼저 train.py를 실행하세요.")
        exit(1)

    # 예측기 생성 및 모델 로드
    predictor = Sim2RealPredictor(model_type="lstm")
    predictor.load_model(str(model_path))

    # 테스트 케이스
    test_cases = [
        {"sim_throughput": 100, "temperature": 25, "humidity": 50,
         "robot_count": 3, "operation_hours": 4, "warehouse_size": 500,
         "description": "정상 조건"},

        {"sim_throughput": 100, "temperature": 35, "humidity": 80,
         "robot_count": 3, "operation_hours": 4, "warehouse_size": 500,
         "description": "고온 + 고습도"},

        {"sim_throughput": 100, "temperature": 25, "humidity": 50,
         "robot_count": 8, "operation_hours": 10, "warehouse_size": 500,
         "description": "로봇 혼잡 + 장시간 가동"},

        {"sim_throughput": 150, "temperature": 20, "humidity": 45,
         "robot_count": 2, "operation_hours": 2, "warehouse_size": 800,
         "description": "이상적 조건"},
    ]

    print("\n예측 테스트:")
    print("-" * 60)

    for case in test_cases:
        desc = case.pop("description")
        result = predictor.predict(**case)

        print(f"\n[{desc}]")
        print(f"  시뮬: {result.sim_throughput} 개/h")
        print(f"  예측: {result.predicted_real_throughput} 개/h")
        print(f"  갭: {result.gap_percent}%")
        print(f"  신뢰도: {result.confidence}")
        print(f"  신뢰구간: {result.confidence_interval}")
        print(f"  요인: {', '.join(result.factors)}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
