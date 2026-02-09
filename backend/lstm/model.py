"""
Forge LSTM Model - Sim2Real 갭 예측 모델
v1.0: PyTorch 기반 LSTM
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """LSTM 모델 설정"""
    # 입력 특성
    input_size: int = 6  # sim_throughput, temp, humidity, robot_count, op_hours, warehouse_size

    # LSTM 구조
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # 출력
    output_size: int = 1  # gap_ratio

    # 시퀀스 설정
    sequence_length: int = 24  # 과거 24시간 참조

    # 학습 설정
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10


class Sim2RealLSTM(nn.Module):
    """
    Sim2Real 갭 예측 LSTM 모델

    입력: [batch, seq_len, features]
    출력: [batch, 1] (갭 비율)
    """

    def __init__(self, config: LSTMConfig = LSTMConfig()):
        super().__init__()
        self.config = config

        # 입력 정규화 레이어
        self.input_norm = nn.BatchNorm1d(config.input_size)

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # 출력 크기 계산
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        # Fully Connected 레이어
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, config.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch, seq_len, features] 또는 [batch, features]

        Returns:
            [batch, output_size]
        """
        # 2D 입력이면 3D로 변환 (단일 타임스텝)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]

        batch_size, seq_len, features = x.shape

        # 입력 정규화 (BatchNorm은 [batch, features] 형태 필요)
        x_reshaped = x.reshape(-1, features)
        x_normed = self.input_norm(x_reshaped)
        x = x_normed.reshape(batch_size, seq_len, features)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 마지막 타임스텝 출력 사용
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]

        # FC 레이어
        output = self.fc(last_output)

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """예측 모드 (eval + no_grad)"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class Sim2RealMLP(nn.Module):
    """
    간단한 MLP 모델 (비시계열용)

    시퀀스 없이 단일 입력으로 예측
    """

    def __init__(self, config: LSTMConfig = LSTMConfig()):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(config.input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(config.dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(config.dropout),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, config.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3D 입력이면 마지막 타임스텝만 사용
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class Sim2RealGRU(nn.Module):
    """
    GRU 모델 (LSTM 대안, 더 빠름)
    """

    def __init__(self, config: LSTMConfig = LSTMConfig()):
        super().__init__()
        self.config = config

        self.input_norm = nn.BatchNorm1d(config.input_size)

        self.gru = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        gru_output_size = config.hidden_size * (2 if config.bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch_size, seq_len, features = x.shape

        x_reshaped = x.reshape(-1, features)
        x_normed = self.input_norm(x_reshaped)
        x = x_normed.reshape(batch_size, seq_len, features)

        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ============================================================
# 모델 팩토리
# ============================================================

def create_model(
    model_type: str = "lstm",
    config: LSTMConfig = None
) -> nn.Module:
    """
    모델 생성 팩토리

    Args:
        model_type: "lstm", "gru", "mlp"
        config: 모델 설정

    Returns:
        모델 인스턴스
    """
    if config is None:
        config = LSTMConfig()

    models = {
        "lstm": Sim2RealLSTM,
        "gru": Sim2RealGRU,
        "mlp": Sim2RealMLP
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](config)


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Forge LSTM Model Test")
    print("=" * 60)

    config = LSTMConfig()

    # 테스트 데이터
    batch_size = 4
    seq_len = 24
    features = 6

    x = torch.randn(batch_size, seq_len, features)

    # LSTM 테스트
    print("\n[1] LSTM 모델")
    lstm_model = Sim2RealLSTM(config)
    output = lstm_model(x)
    print(f"  입력: {x.shape}")
    print(f"  출력: {output.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in lstm_model.parameters()):,}")

    # GRU 테스트
    print("\n[2] GRU 모델")
    gru_model = Sim2RealGRU(config)
    output = gru_model(x)
    print(f"  입력: {x.shape}")
    print(f"  출력: {output.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in gru_model.parameters()):,}")

    # MLP 테스트 (2D 입력)
    print("\n[3] MLP 모델")
    x_2d = torch.randn(batch_size, features)
    mlp_model = Sim2RealMLP(config)
    output = mlp_model(x_2d)
    print(f"  입력: {x_2d.shape}")
    print(f"  출력: {output.shape}")
    print(f"  파라미터 수: {sum(p.numel() for p in mlp_model.parameters()):,}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
