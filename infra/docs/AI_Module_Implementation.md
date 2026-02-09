# Forge v2.0 AI Module 구현 상세

## 1. LLM Module - 자연어 → Isaac Sim 파라미터 변환

### 1.1 왜 LLM이 필요한가?

**문제 상황:**
```
사용자 요청: "로봇 50대가 들어갈 수 있는 대형 창고에서
            AGV 5대와 포크리프트 2대를 배치하고,
            선반은 4열로 구성해줘"
```

**규칙 기반(Rule-based)으로는 불가능한 이유:**

| 요청 표현 | 필요한 추론 |
|----------|------------|
| "50대가 들어갈 수 있는" | 창고 크기 계산 (50대 × 100m² = 5,000m² → 70m × 70m) |
| "대형 창고" | 문맥에 따른 크기 조정 |
| "AGV 5대와 포크리프트 2대" | 다중 로봇 타입 파싱, 각각 다른 파라미터 적용 |
| "선반 4열" | count=4, arrangement="row" 매핑 |

**규칙으로 처리하려면:**
- 수백 개의 패턴 매칭 규칙 필요
- 새로운 표현마다 코드 수정 필요
- "여유있게", "빽빽하게" 같은 추상적 표현 처리 불가

**LLM으로 해결:**
- 자연어 이해 + 추론 능력
- 문맥에 따른 유연한 해석
- 새로운 표현도 학습 없이 대응

---

### 1.2 추론 가이드라인 (prompts.py)

LLM에게 제공하는 추론 규칙:

```python
## 추론 가이드라인 (중요!)

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
- "습한 환경" → humidity: 80
```

---

### 1.3 변환 예시

**입력:**
```
"로봇 50대가 들어갈 수 있는 대형 창고에서
AGV 5대와 포크리프트 2대를 배치하고, 선반은 4열로 구성해줘"
```

**LLM 추론 과정:**
1. "50대가 들어갈 수 있는" → 50 × 100m² = 5,000m² → √5000 ≈ 70m
2. "대형 창고" 확인 → 70m × 70m (대형 기준 충족)
3. "AGV 5대" → robots[0]: {type: "AGV", count: 5, speed: 1.5}
4. "포크리프트 2대" → robots[1]: {type: "forklift", count: 2, speed: 2.0}
5. "선반 4열" → obstacles[0]: {type: "shelf", count: 4, arrangement: "row"}

**출력 (JSON):**
```json
{
  "environment": {
    "type": "warehouse",
    "width": 70,
    "length": 70,
    "height": 8
  },
  "robots": [
    {"type": "AGV", "count": 5, "speed": 1.5},
    {"type": "forklift", "count": 2, "speed": 2.0}
  ],
  "obstacles": [
    {"type": "shelf", "count": 4, "arrangement": "row", "spacing": 3.0}
  ]
}
```

---

### 1.4 민감도 기반 LLM 분기

| 민감도 | 데이터 예시 | 사용 LLM |
|--------|------------|----------|
| PUBLIC | 일반 환경 설정 | OpenAI (빠름) |
| INTERNAL | 시뮬레이션 결과, 생산량 | Qwen 로컬 (보안) |
| CONFIDENTIAL | 고객사 정보, 비밀번호 | Qwen 로컬 (보안) |

---

### 1.5 주요 기능

| 함수 | 역할 | 입력 | 출력 |
|------|------|------|------|
| `parse_environment_request()` | 자연어 → JSON | "30m 창고에 AGV 3대" | Isaac Sim 파라미터 |
| `modify_parameters()` | 대화형 수정 | "속도 올려줘" | 수정된 JSON |
| `analyze_simulation_result()` | 결과 분석 | 시뮬 로그 | 인사이트 텍스트 |
| `generate_report_from_lstm()` | 보고서 생성 | LSTM 예측값 | 자연어 보고서 |

---

## 2. LSTM Module - Sim2Real 갭 예측

### 2.1 왜 LSTM이 필요한가?

**문제 상황:**
```
시뮬레이션 결과: 100개/시간
실제 현장:        ???
```

**Sim2Real Gap (시뮬레이션 ↔ 현실 격차):**
- 시뮬레이션은 이상적 환경
- 현실에는 온도, 습도, 장비 노후화 등 변수 존재
- 동일 설정이어도 현장 성능은 10~30% 낮을 수 있음

**규칙으로 처리하려면:**
- 각 환경 변수마다 공식 필요
- 변수 간 상호작용 모델링 불가능
- 축적된 데이터 패턴 활용 불가

**LSTM으로 해결:**
- 시계열 패턴 학습 (과거 데이터 기반)
- 복합 변수 간 상호작용 자동 학습
- 신뢰구간 제공 (불확실성 정량화)

---

### 2.2 모델 구조 (v3 - 현재 사용 중)

**사용 중인 모델:** `lstm/data/sim2real_lstm_v3.pth`

| 항목 | 값 |
|------|-----|
| 모델 타입 | LSTM (2 layers) |
| Hidden Size | 64 |
| 입력 특성 | 10개 |
| 시퀀스 길이 | 24 (24시간) |
| 학습 데이터 | 10년 (87,600 샘플) |
| Train R² | 0.825 |
| Val R² | 0.822 |
| **Test R²** | **0.826** |

---

### 2.3 입력 특성 (10개)

| 특성 | 설명 | 영향 |
|------|------|------|
| `sim_throughput` | 시뮬레이션 처리량 (개/h) | 기준값 |
| `temperature` | 현장 온도 (°C) | 고온 시 성능↓ |
| `humidity` | 현장 습도 (%) | 고습 시 성능↓ |
| `robot_count` | 로봇 대수 | 과밀 시 성능↓ |
| `operation_hours` | 연속 가동 시간 | 장시간 시 피로↑ |
| `days_since_maintenance` | 정비 후 경과일 | 노후화 영향 |
| `hour` | 시간대 (0-23) | 일간 패턴 |
| `day_of_week` | 요일 (0-6) | 주간 패턴 |
| `is_weekend` | 주말 여부 (0/1) | 운영 패턴 |
| `day_of_year` | 연중 일수 (0-365) | 계절성 |

---

### 2.4 출력 값

```json
{
  "sim_throughput": 100,
  "predicted_real_throughput": 73.26,
  "gap_ratio": -0.2674,
  "gap_percent": -26.74,
  "confidence": 0.83,
  "confidence_interval": [71.1, 75.5],
  "factors": ["고온 (32°C)", "고습도 (75%)"]
}
```

| 출력 | 설명 |
|------|------|
| `predicted_real_throughput` | 예상 실제 처리량 |
| `gap_percent` | 시뮬 대비 격차 (%) |
| `confidence` | 예측 신뢰도 (0~1) |
| `confidence_interval` | 95% 신뢰구간 |
| `factors` | 주요 영향 요인 |

---

### 2.5 영향 요인 분석

```python
def _analyze_factors(self, temperature, humidity, robot_count, operation_hours):
    factors = []

    if temperature > 30:  # 고온
        factors.append(f"고온 ({temperature}°C)")

    if humidity > 70:  # 고습도
        factors.append(f"고습도 ({humidity}%)")

    if robot_count > 6:  # 혼잡
        factors.append(f"로봇 혼잡 ({robot_count}대)")

    if operation_hours > 8:  # 장시간 가동
        factors.append(f"장시간 가동 ({operation_hours}시간)")

    return factors if factors else ["정상 조건"]
```

---

### 2.6 LSTM → LLM 파이프라인 (v2.0 핵심)

**기존 (v1.0):**
```
LSTM 출력: {gap: -26.74%, confidence: 0.83, factors: [...]}
→ 사람이 해석해야 함
```

**개선 (v2.0):**
```
LSTM 출력 → LLM → 자연어 보고서

"현장 적용 시 약 27% 성능 저하가 예상됩니다. (신뢰도 83%)
주요 원인은 고온과 고습도입니다.
권장: 환기 시스템 점검 후 적용하세요."
```

---

## 3. 현재 설정 (완료)

### LSTM 모델 경로

**api/main.py:**
```python
# v3 모델 경로 (우선)
v3_model_path = Path(__file__).parent.parent / "lstm" / "data" / "sim2real_lstm_v3.pth"
# 기존 모델 경로 (fallback)
legacy_model_path = Path(__file__).parent.parent / "lstm" / "checkpoints" / "mlp_sim2real_best.pt"
```

**predict.py:**
```python
# v3 모델 경로 (우선)
v3_model_path = Path(__file__).parent / "data" / "sim2real_lstm_v3.pth"
```

✅ **v3 모델로 변경 완료**

---

## 4. 요약

| 모듈 | 역할 | 핵심 기능 | 왜 AI? |
|------|------|----------|--------|
| **LLM** | 자연어 이해 | 추론 기반 파라미터 변환 | 규칙으로 불가능한 유연한 해석 |
| **LSTM** | 갭 예측 | 환경 조건 → 실제 성능 예측 | 복합 변수 상호작용 학습 |
| **LLM+LSTM** | 보고서 생성 | 숫자 → 자연어 보고서 | 비전문가도 이해 가능 |

### 테스트 결과 (v3 모델)

```
입력:
  sim_throughput: 100
  temperature: 32°C
  humidity: 75%
  robot_count: 5
  operation_hours: 6

출력:
  predicted_real_throughput: 73.26 개/h
  gap_percent: -26.74%
  factors: ["고온 (32.0°C)", "고습도 (75.0%)"]
```
