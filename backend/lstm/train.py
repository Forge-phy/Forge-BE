"""
Forge LSTM Training - 모델 학습 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import json
from datetime import datetime

from model import LSTMConfig, create_model


# ============================================================
# 데이터셋 클래스
# ============================================================

class Sim2RealDataset(Dataset):
    """비시계열 데이터셋 (MLP용)"""

    FEATURE_COLS = [
        'sim_throughput', 'temperature', 'humidity',
        'robot_count', 'operation_hours', 'warehouse_size'
    ]
    TARGET_COL = 'gap_ratio'

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = None,
        normalize: bool = True
    ):
        self.feature_cols = feature_cols or self.FEATURE_COLS
        self.target_col = target_col or self.TARGET_COL

        # 특성과 타겟 분리
        self.X = data[self.feature_cols].values.astype(np.float32)
        self.y = data[self.target_col].values.astype(np.float32).reshape(-1, 1)

        # 정규화
        self.normalize = normalize
        if normalize:
            self.X_mean = self.X.mean(axis=0)
            self.X_std = self.X.std(axis=0) + 1e-8
            self.X = (self.X - self.X_mean) / self.X_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    def get_normalization_params(self) -> Dict:
        """정규화 파라미터 반환 (추론 시 필요)"""
        if self.normalize:
            return {
                'mean': self.X_mean.tolist(),
                'std': self.X_std.tolist()
            }
        return None


class Sim2RealSequenceDataset(Dataset):
    """시계열 데이터셋 (LSTM/GRU용)"""

    FEATURE_COLS = [
        'sim_throughput', 'temperature', 'humidity',
        'robot_count', 'operation_hours', 'warehouse_size'
    ]
    TARGET_COL = 'gap_ratio'

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 24,
        feature_cols: List[str] = None,
        target_col: str = None,
        normalize: bool = True
    ):
        self.feature_cols = feature_cols or self.FEATURE_COLS
        self.target_col = target_col or self.TARGET_COL
        self.sequence_length = sequence_length

        # 특성과 타겟
        X = data[self.feature_cols].values.astype(np.float32)
        y = data[self.target_col].values.astype(np.float32)

        # 정규화
        self.normalize = normalize
        if normalize:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
            X = (X - self.X_mean) / self.X_std

        # 시퀀스 생성
        self.sequences = []
        self.targets = []

        for i in range(len(X) - sequence_length):
            self.sequences.append(X[i:i + sequence_length])
            self.targets.append(y[i + sequence_length])

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets).reshape(-1, 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.targets[idx])
        )

    def get_normalization_params(self) -> Dict:
        if self.normalize:
            return {
                'mean': self.X_mean.tolist(),
                'std': self.X_std.tolist()
            }
        return None


# ============================================================
# 학습 함수
# ============================================================

@dataclass
class TrainingResult:
    """학습 결과"""
    model_path: str
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float
    training_time: float
    config: dict


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LSTMConfig,
    save_dir: str = None,
    model_name: str = "sim2real_model"
) -> TrainingResult:
    """
    모델 학습

    Args:
        model: PyTorch 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        config: 학습 설정
        save_dir: 모델 저장 디렉토리
        model_name: 모델 이름

    Returns:
        TrainingResult
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = model.to(device)

    # 손실 함수 & 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    start_time = datetime.now()

    for epoch in range(config.epochs):
        # ===== Training =====
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 학습률 조정
        scheduler.step(val_loss)

        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # 최고 모델 저장
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / f"{model_name}_best.pt")
        else:
            patience_counter += 1

        # 로그 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} (Epoch {best_epoch+1})")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    training_time = (datetime.now() - start_time).total_seconds()

    # 최종 모델 저장
    if save_dir:
        save_path = Path(save_dir)

        # 마지막 모델
        torch.save(model.state_dict(), save_path / f"{model_name}_final.pt")

        # 설정 저장
        config_dict = {
            'input_size': config.input_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'output_size': config.output_size,
            'sequence_length': config.sequence_length,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'training_time': training_time
        }
        with open(save_path / f"{model_name}_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        model_path = str(save_path / f"{model_name}_best.pt")
    else:
        model_path = None

    return TrainingResult(
        model_path=model_path,
        train_loss=train_losses,
        val_loss=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        training_time=training_time,
        config=config_dict if save_dir else {}
    )


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None
) -> Dict:
    """
    모델 평가

    Returns:
        평가 지표 딕셔너리
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)

            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.numpy().flatten())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 평가 지표 계산
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100

    # R² Score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'n_samples': len(actuals)
    }


# ============================================================
# 메인 학습 실행
# ============================================================

def main():
    print("=" * 60)
    print("Forge LSTM Training")
    print("=" * 60)

    # 설정
    config = LSTMConfig(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        early_stopping_patience=15
    )

    # 데이터 로드
    data_dir = Path(__file__).parent / "data"
    save_dir = Path(__file__).parent / "checkpoints"

    print("\n[1] 데이터 로드...")

    # 기본 데이터셋 (비시계열)
    df_basic = pd.read_csv(data_dir / "sim2real_basic.csv")
    print(f"  기본 데이터: {len(df_basic)} 샘플")

    # 시계열 데이터셋
    df_timeseries = pd.read_csv(data_dir / "sim2real_timeseries.csv")
    print(f"  시계열 데이터: {len(df_timeseries)} 샘플")

    # ===== MLP 모델 학습 (비시계열) =====
    print("\n[2] MLP 모델 학습 (비시계열 데이터)...")

    dataset_mlp = Sim2RealDataset(df_basic)
    train_size = int(0.8 * len(dataset_mlp))
    val_size = len(dataset_mlp) - train_size

    train_dataset, val_dataset = random_split(dataset_mlp, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    mlp_model = create_model("mlp", config)
    print(f"  파라미터 수: {sum(p.numel() for p in mlp_model.parameters()):,}")

    mlp_result = train_model(
        mlp_model, train_loader, val_loader, config,
        save_dir=str(save_dir), model_name="mlp_sim2real"
    )

    print(f"\n  최고 검증 손실: {mlp_result.best_val_loss:.6f} (Epoch {mlp_result.best_epoch+1})")
    print(f"  학습 시간: {mlp_result.training_time:.1f}초")

    # MLP 평가
    mlp_metrics = evaluate_model(mlp_model, val_loader)
    print(f"  평가 결과:")
    print(f"    - RMSE: {mlp_metrics['rmse']:.4f}")
    print(f"    - MAE: {mlp_metrics['mae']:.4f}")
    print(f"    - R²: {mlp_metrics['r2']:.4f}")

    # 정규화 파라미터 저장
    norm_params = dataset_mlp.get_normalization_params()
    with open(save_dir / "mlp_sim2real_norm.json", 'w') as f:
        json.dump(norm_params, f, indent=2)

    # ===== LSTM 모델 학습 (시계열) =====
    print("\n[3] LSTM 모델 학습 (시계열 데이터)...")

    config.sequence_length = 12  # 12시간 시퀀스

    dataset_lstm = Sim2RealSequenceDataset(df_timeseries, sequence_length=config.sequence_length)
    train_size = int(0.8 * len(dataset_lstm))
    val_size = len(dataset_lstm) - train_size

    train_dataset, val_dataset = random_split(dataset_lstm, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    lstm_model = create_model("lstm", config)
    print(f"  시퀀스 길이: {config.sequence_length}")
    print(f"  파라미터 수: {sum(p.numel() for p in lstm_model.parameters()):,}")

    lstm_result = train_model(
        lstm_model, train_loader, val_loader, config,
        save_dir=str(save_dir), model_name="lstm_sim2real"
    )

    print(f"\n  최고 검증 손실: {lstm_result.best_val_loss:.6f} (Epoch {lstm_result.best_epoch+1})")
    print(f"  학습 시간: {lstm_result.training_time:.1f}초")

    # LSTM 평가
    lstm_metrics = evaluate_model(lstm_model, val_loader)
    print(f"  평가 결과:")
    print(f"    - RMSE: {lstm_metrics['rmse']:.4f}")
    print(f"    - MAE: {lstm_metrics['mae']:.4f}")
    print(f"    - R²: {lstm_metrics['r2']:.4f}")

    # 정규화 파라미터 저장
    norm_params = dataset_lstm.get_normalization_params()
    with open(save_dir / "lstm_sim2real_norm.json", 'w') as f:
        json.dump(norm_params, f, indent=2)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"\n저장된 모델:")
    print(f"  - {save_dir}/mlp_sim2real_best.pt")
    print(f"  - {save_dir}/lstm_sim2real_best.pt")


if __name__ == "__main__":
    main()
