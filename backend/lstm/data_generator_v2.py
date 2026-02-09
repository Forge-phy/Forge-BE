"""
Forge LSTM - 가상 Sim2Real 갭 데이터 생성기 v2.0
LSTM 학습에 적합하게 재설계

핵심 변경:
1. 시계열 패턴 추가 (추세, 주기성, 자기상관)
2. 비선형 관계 추가
3. 환경 조건의 누적 효과
4. 현실적인 공장 시나리오 반영
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import json
from pathlib import Path


def generate_realistic_timeseries(
    n_days: int = 90,
    samples_per_day: int = 24,
    seed: int = 42
) -> pd.DataFrame:
    """
    LSTM 학습에 적합한 현실적 시계열 데이터 생성

    특징:
    - 장기 추세 (계절성, 설비 노후화)
    - 일간/주간 주기성
    - 환경 조건의 누적/지연 효과
    - 비선형 상호작용
    - 현실적인 노이즈
    """
    np.random.seed(seed)

    total_samples = n_days * samples_per_day
    records = []

    # ========== 장기 추세 설정 ==========
    # 설비 노후화: 시간이 지날수록 효율 감소 (비선형)
    aging_factor = np.linspace(0, 1, total_samples)
    aging_effect = -0.05 * (aging_factor ** 1.5)  # 초반 적고 후반 급증

    # 계절 효과 (90일 = 약 3개월, 여름→가을 전환 가정)
    seasonal_base_temp = 30 - 8 * np.linspace(0, 1, total_samples)  # 30°C → 22°C

    # ========== 공장 기본 설정 ==========
    base_sim_throughput = 100  # 기본 시뮬레이션 처리량
    warehouse_size = 1000  # 고정 (같은 공장)

    # ========== 상태 변수 (누적 효과용) ==========
    cumulative_heat_stress = 0  # 누적 열 스트레스
    cumulative_fatigue = 0  # 누적 피로도
    maintenance_day = 30  # 30일마다 정비

    prev_gap = -0.10  # 이전 갭 (자기상관용)

    for day in range(n_days):
        # 주간 패턴: 주말(토,일)은 다른 패턴
        day_of_week = day % 7
        is_weekend = day_of_week >= 5

        # 정비 효과: 정비 후 효율 회복
        days_since_maintenance = day % maintenance_day
        maintenance_effect = 0.02 * (1 - days_since_maintenance / maintenance_day)

        # 일별 로봇 수 (주중 4-6대, 주말 2-3대)
        if is_weekend:
            robot_count = np.random.randint(2, 4)
            daily_target = 80  # 주말 목표 낮음
        else:
            robot_count = np.random.randint(4, 7)
            daily_target = 100

        for hour in range(samples_per_day):
            idx = day * samples_per_day + hour

            # ========== 환경 조건 생성 ==========

            # 온도: 계절 + 일간 패턴 + 노이즈
            base_temp = seasonal_base_temp[idx]
            hour_temp_variation = 6 * np.sin((hour - 6) * np.pi / 12)  # 낮에 높음
            temperature = base_temp + hour_temp_variation + np.random.normal(0, 1.5)
            temperature = np.clip(temperature, 15, 40)

            # 습도: 온도와 역상관 + 노이즈
            base_humidity = 70 - 0.8 * (temperature - 25)
            humidity = base_humidity + np.random.normal(0, 5)
            humidity = np.clip(humidity, 30, 90)

            # 가동 시간: 8시간 근무 패턴
            if hour in [8, 16]:  # 교대 시간
                cumulative_fatigue = 0
            operation_hours = min(8, (hour % 8) + 1) if 6 <= hour <= 22 else 0

            # ========== 시뮬레이션 처리량 ==========
            # 시간대별 변동 + 약간의 노이즈
            if 6 <= hour <= 22:  # 주간 가동
                hour_efficiency = 1.0 - 0.1 * abs(hour - 14) / 8  # 오후 2시 피크
                sim_throughput = daily_target * hour_efficiency
                sim_throughput += np.random.normal(0, 3)
                sim_throughput = max(50, sim_throughput)
            else:  # 야간 (저가동)
                sim_throughput = 30 + np.random.normal(0, 5)
                sim_throughput = max(20, sim_throughput)

            # ========== Sim2Real Gap 계산 (비선형, 상호작용) ==========

            # 1. 기본 갭
            gap = -0.08

            # 2. 온도 효과 (비선형: 25°C 기준, 멀어질수록 급증)
            temp_deviation = abs(temperature - 25)
            temp_effect = -0.003 * (temp_deviation ** 1.3)

            # 3. 습도 효과 (60% 이상에서 급격히 나빠짐)
            if humidity > 60:
                humidity_effect = -0.004 * ((humidity - 60) ** 1.2)
            else:
                humidity_effect = 0.001 * (60 - humidity) / 30  # 건조하면 약간 좋음

            # 4. 온도-습도 상호작용 (고온다습 = 최악)
            if temperature > 28 and humidity > 65:
                interaction_effect = -0.02 * ((temperature - 28) / 10) * ((humidity - 65) / 25)
            else:
                interaction_effect = 0

            # 5. 로봇 혼잡도 (비선형)
            if robot_count > 4:
                congestion_effect = -0.015 * ((robot_count - 4) ** 1.5)
            else:
                congestion_effect = 0

            # 6. 피로도 효과 (누적)
            cumulative_fatigue += 0.002 * operation_hours
            fatigue_effect = -cumulative_fatigue * 0.5
            fatigue_effect = max(-0.05, fatigue_effect)  # 최대 -5%

            # 7. 열 스트레스 누적 (온도 28도 이상일 때)
            if temperature > 28:
                cumulative_heat_stress += 0.001 * (temperature - 28)
            else:
                cumulative_heat_stress = max(0, cumulative_heat_stress - 0.002)
            heat_stress_effect = -cumulative_heat_stress * 0.3
            heat_stress_effect = max(-0.08, heat_stress_effect)

            # 8. 설비 노후화 (장기 추세)
            aging = aging_effect[idx]

            # 9. 정비 효과
            maint = maintenance_effect

            # 10. 자기상관 (이전 갭의 영향)
            autocorr_effect = 0.3 * (prev_gap + 0.10)  # 이전 갭이 나빴으면 현재도 나쁨

            # 11. 노이즈
            noise = np.random.normal(0, 0.015)

            # 최종 갭 계산
            gap = (gap + temp_effect + humidity_effect + interaction_effect +
                   congestion_effect + fatigue_effect + heat_stress_effect +
                   aging + maint + autocorr_effect + noise)

            # 갭 범위 제한
            gap = np.clip(gap, -0.35, 0.05)

            # 자기상관용 저장
            prev_gap = gap

            # ========== 실제 처리량 계산 ==========
            real_throughput = sim_throughput * (1 + gap)
            real_throughput = max(0, real_throughput)

            # ========== 레코드 저장 ==========
            records.append({
                'day': day,
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': int(is_weekend),
                'timestamp': idx,
                'sim_throughput': round(sim_throughput, 2),
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'robot_count': robot_count,
                'operation_hours': operation_hours,
                'days_since_maintenance': days_since_maintenance,
                'real_throughput': round(real_throughput, 2),
                'gap_ratio': round(gap, 4),
                'gap_percent': round(gap * 100, 2)
            })

    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, filename: str, output_dir: str = None):
    """데이터셋 저장"""
    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    csv_path = output_dir / f"{filename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV 저장: {csv_path}")

    # 통계 정보 저장
    stats = {
        "n_samples": len(df),
        "n_days": df['day'].nunique(),
        "columns": list(df.columns),
        "statistics": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            for col in df.select_dtypes(include=[np.number]).columns
        }
    }

    stats_path = output_dir / f"{filename}_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"통계 저장: {stats_path}")

    return csv_path


if __name__ == "__main__":
    print("=" * 60)
    print("Forge LSTM - 가상 Sim2Real 데이터셋 생성 v2.0")
    print("=" * 60)

    # 90일 시계열 데이터 생성 (2160 샘플)
    print("\n[1] 시계열 데이터셋 생성 (90일 x 24시간)...")
    df = generate_realistic_timeseries(n_days=90, seed=42)
    save_dataset(df, "sim2real_v2")

    print(f"\n데이터 shape: {df.shape}")
    print(f"\n샘플 데이터 (Day 0-1):")
    print(df[df['day'] <= 1].head(10).to_string())

    print(f"\n=== 주요 통계 ===")
    print(f"Gap 평균: {df['gap_percent'].mean():.2f}%")
    print(f"Gap 범위: {df['gap_percent'].min():.2f}% ~ {df['gap_percent'].max():.2f}%")
    print(f"Gap 표준편차: {df['gap_percent'].std():.2f}%")

    print(f"\n=== 상관관계 ===")
    corr_cols = ['sim_throughput', 'temperature', 'humidity',
                 'robot_count', 'operation_hours', 'real_throughput', 'gap_percent']
    print(df[corr_cols].corr()['gap_percent'].round(3))

    print("\n" + "=" * 60)
    print("데이터셋 생성 완료!")
    print("=" * 60)
