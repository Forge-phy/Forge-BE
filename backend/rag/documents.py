"""
Forge RAG Documents - Isaac Sim 전문 문서 및 예시
"""

# ============================================================
# Isaac Sim 공식 문서 기반 지식
# ============================================================

ISAAC_SIM_DOCUMENTS = [
    # 환경 설정 관련
    {
        "text": """## Isaac Sim 환경 유형 가이드

### Warehouse (창고)
- 용도: 물류, 배송 센터, 보관 시설
- 특징: 선반 배치, 통로 최적화, 피킹 작업
- 권장 설정:
  - 천장 높이: 5-15m
  - 통로 폭: 로봇 폭의 2배 이상
  - 조명: 균일 분포

### Factory (공장)
- 용도: 제조, 조립, 가공 시설
- 특징: 생산라인, 작업대, 컨베이어
- 권장 설정:
  - 천장 높이: 6-20m
  - 작업 영역: 로봇 도달 범위 고려
  - 안전 구역: 필수 설정

### Custom (커스텀)
- 용도: 특수 환경, 연구용
- 특징: 자유로운 설정
- 주의: 물리적 제약 직접 설정 필요""",
        "metadata": {"source": "isaac_sim_docs", "topic": "environment_types", "importance": "high"}
    },
    {
        "text": """## Isaac Sim 좌표계 및 단위

### 좌표계
- X축: 동서 방향 (양수 = 동쪽)
- Y축: 남북 방향 (양수 = 북쪽)
- Z축: 높이 (양수 = 위)
- 원점: 환경 중심 또는 좌측 하단

### 단위 시스템
- 거리: 미터 (m)
- 각도: 라디안 (rad) 또는 도 (deg)
- 시간: 초 (s)
- 질량: 킬로그램 (kg)
- 힘: 뉴턴 (N)
- 속도: m/s
- 가속도: m/s²

### 스케일 가이드
- 소형 환경: 10x10m ~ 50x50m
- 중형 환경: 50x50m ~ 200x200m
- 대형 환경: 200x200m ~ 1000x1000m
- 초대형: 1000m+ (성능 주의)""",
        "metadata": {"source": "isaac_sim_docs", "topic": "coordinates", "importance": "high"}
    },

    # 로봇 관련
    {
        "text": """## AGV (Automated Guided Vehicle) 상세 가이드

### 특성
- 고정 경로 추종
- 자기 테이프, QR 코드, 레이저 가이드 사용
- 단순한 내비게이션, 높은 신뢰성

### 권장 파라미터
- 속도: 0.5-2.0 m/s (일반), 최대 3.0 m/s
- 가속도: 0.5-1.5 m/s²
- 적재량: 50-500 kg (일반), 최대 2000 kg
- 배터리: 8-12시간 운영
- 충전 시간: 1-3시간 (급속 30분)

### 충돌 회피
- 기본 전방 센서: 2-5m
- 측면 센서: 선택적
- 비상 정지 거리: 속도 × 0.5초 + 0.3m

### 경로 설정
- 직선 구간: 효율적
- 곡선 반경: 최소 회전 반경 × 1.2
- 교차로: 우선순위 규칙 필수""",
        "metadata": {"source": "isaac_sim_docs", "topic": "agv", "importance": "high"}
    },
    {
        "text": """## AMR (Autonomous Mobile Robot) 상세 가이드

### 특성
- 자율 경로 계획
- SLAM 기반 내비게이션
- 동적 장애물 회피

### 권장 파라미터
- 속도: 1.0-3.0 m/s (일반), 최대 5.0 m/s
- 가속도: 1.0-2.0 m/s²
- 적재량: 50-300 kg (협동), 최대 1500 kg
- 센서: LiDAR + 카메라 + IMU

### 내비게이션 설정
- 지도 해상도: 0.05-0.1 m/pixel
- 경로 재계획 주기: 0.5-2.0초
- 장애물 팽창 반경: 로봇 반경 + 0.1m

### 다중 로봇 조율
- 트래픽 관리자 필요
- 우선순위 기반 또는 경매 기반
- 데드락 방지 알고리즘 권장""",
        "metadata": {"source": "isaac_sim_docs", "topic": "amr", "importance": "high"}
    },
    {
        "text": """## 로봇 암 (Manipulator) 가이드

### 유형
- 6축 산업용 로봇
- 협동 로봇 (Cobot)
- 스카라 (SCARA)
- 델타 로봇

### 권장 파라미터
- 작업 반경: 0.5-3.0 m
- 적재량: 1-50 kg
- 반복 정밀도: 0.01-0.1 mm
- 최대 속도: 관절별 다름

### 안전 설정
- 속도 제한 영역 설정
- 힘/토크 제한
- 인간 감지 센서 연동

### Isaac Sim 특수 설정
- 관절 제한 (joint limits)
- 충돌 메쉬 단순화
- 자가 충돌 체크""",
        "metadata": {"source": "isaac_sim_docs", "topic": "robot_arm", "importance": "medium"}
    },

    # 장애물/환경 요소
    {
        "text": """## 선반 (Shelf) 배치 가이드

### 배치 유형
1. **Row (열 배치)**
   - 평행한 열로 배치
   - 통로 효율성 높음
   - 피킹 작업에 적합

2. **Grid (격자 배치)**
   - 균일한 간격의 격자
   - 접근성 균일
   - 대형 창고에 적합

3. **Random (랜덤)**
   - 불규칙 배치
   - 테스트/연구용
   - 실제 환경 모사

### 권장 간격
- 통로 폭: 로봇 폭 × 2 + 0.5m
- 선반 간 거리: 최소 1.5m
- 벽과의 거리: 최소 1.0m

### 선반 크기 (일반적)
- 소형: 0.5m × 1.0m × 2.0m
- 중형: 1.0m × 2.0m × 2.5m
- 대형: 1.5m × 3.0m × 4.0m""",
        "metadata": {"source": "isaac_sim_docs", "topic": "shelf", "importance": "medium"}
    },
    {
        "text": """## 컨베이어 (Conveyor) 설정 가이드

### 유형
- 벨트 컨베이어: 일반 물품 이송
- 롤러 컨베이어: 중량물, 팔레트
- 체인 컨베이어: 고온, 특수 환경

### 파라미터
- 속도: 0.1-2.0 m/s
- 폭: 0.3-2.0 m
- 높이: 0.5-1.2 m (작업 높이)
- 적재 용량: 10-500 kg/m

### 로봇 연동
- 컨베이어 ↔ 로봇 인터페이스 필요
- 동기화 신호 설정
- 버퍼 영역 확보

### 레이아웃 팁
- 로봇 동선과 최소 교차
- 곡선부 속도 감소 고려
- 합류/분기점 주의""",
        "metadata": {"source": "isaac_sim_docs", "topic": "conveyor", "importance": "medium"}
    },

    # 시뮬레이션 설정
    {
        "text": """## 시뮬레이션 시간 설정 가이드

### 시간 간격 (time_step)
- 0.001초: 고정밀 (로봇 암, 충돌 분석)
- 0.01초: 표준 (대부분의 시뮬레이션)
- 0.1초: 저정밀 (장시간 시뮬레이션)

### 시뮬레이션 시간 (duration)
- 300초 (5분): 빠른 테스트
- 3600초 (1시간): 표준 분석
- 28800초 (8시간): 1교대 시뮬레이션
- 86400초 (24시간): 1일 분석

### 실시간 배율 (realtime_factor)
- 0.1: 슬로모션 (디버깅)
- 1.0: 실시간
- 10.0: 가속 (빠른 결과 확인)
- 100.0: 최대 가속 (장시간 시뮬)

### 성능 고려사항
- 로봇 수 × 장애물 수 = 계산 복잡도
- time_step 감소 → 정확도↑, 속도↓
- 대규모 환경: GPU 가속 권장""",
        "metadata": {"source": "isaac_sim_docs", "topic": "simulation_time", "importance": "high"}
    },
    {
        "text": """## 물리 엔진 설정 가이드

### PhysX 설정
- GPU 가속: 대규모 시뮬레이션 필수
- 솔버 반복: 4-16 (정확도 vs 속도)
- 접촉 오프셋: 0.001-0.01 m

### 충돌 감지
- 연속 충돌 감지 (CCD): 고속 물체
- 이산 충돌 감지: 일반 물체
- 충돌 그룹: 최적화용 분류

### 마찰 설정
- 바닥 마찰: 0.5-0.8 (일반)
- 고무 바퀴: 0.7-0.9
- 미끄러운 바닥: 0.3-0.5

### 중력
- 기본: -9.81 m/s² (Z축)
- 달/화성: 별도 설정
- 무중력: 특수 환경""",
        "metadata": {"source": "isaac_sim_docs", "topic": "physics", "importance": "medium"}
    },

    # 최적화 및 분석
    {
        "text": """## 시뮬레이션 최적화 전략

### 충돌 문제 해결
1. **교차로 충돌**
   - 원인: 동시 도착, 우선순위 없음
   - 해결: 시간차 출발 (1-3초 간격)
   - 해결: 교차로 예약 시스템

2. **좁은 통로 충돌**
   - 원인: 양방향 통행
   - 해결: 일방통행 설정
   - 해결: 대기 구역 추가

3. **병목 현상**
   - 원인: 작업 집중, 경로 제한
   - 해결: 작업 분산
   - 해결: 대체 경로 추가

### 처리량 최적화
- 로봇 속도 증가 (안전 범위 내)
- 경로 최적화 (최단 경로)
- 작업 할당 알고리즘 개선
- 충전 전략 최적화

### 에너지 효율
- 불필요한 이동 최소화
- 대기 시 저전력 모드
- 충전 스케줄 최적화""",
        "metadata": {"source": "isaac_sim_docs", "topic": "optimization", "importance": "high"}
    },
    {
        "text": """## Sim2Real 갭 완화 가이드

### 주요 갭 원인
1. **센서 노이즈**
   - 시뮬: 완벽한 센서
   - 현실: 노이즈, 드리프트, 간섭
   - 완화: 시뮬에 노이즈 모델 추가

2. **바닥 상태**
   - 시뮬: 균일한 마찰
   - 현실: 얼룩, 경사, 이물질
   - 완화: 마찰 변동 추가 (±20%)

3. **환경 조건**
   - 시뮬: 고정 온도/습도
   - 현실: 변동, 계절 영향
   - 완화: 환경 파라미터 범위 설정

4. **예상치 못한 장애물**
   - 시뮬: 정적 환경
   - 현실: 사람, 임시 물체
   - 완화: 동적 장애물 추가

### 갭 예측 공식
- 현장 성능 ≈ 시뮬 성능 × (0.80 ~ 0.95)
- 갭 크기: 10-25% (일반적)
- 고온/고습: 추가 5-10% 감소
- 노후 장비: 추가 5-15% 감소

### 신뢰구간
- 90% 신뢰: 시뮬 × (0.75, 0.98)
- 80% 신뢰: 시뮬 × (0.80, 0.95)""",
        "metadata": {"source": "isaac_sim_docs", "topic": "sim2real", "importance": "high"}
    },
    {
        "text": """## 결과 분석 메트릭 가이드

### 핵심 KPI
1. **처리량 (Throughput)**
   - 단위: items/hour, pallets/hour
   - 목표: 비즈니스 요구사항 충족
   - 벤치마크: 기존 시스템 대비

2. **가동률 (Utilization)**
   - 정의: 실제 작업 시간 / 전체 시간
   - 목표: 70-85%
   - 100%는 비현실적 (충전, 대기 필요)

3. **효율성 (Efficiency)**
   - 정의: 이론적 최대 / 실제 성능
   - 목표: 80-95%
   - 병목 식별 지표

### 분석 지표
- 평균 작업 완료 시간
- 대기 시간 비율
- 충돌 빈도 및 위치
- 에너지 소비량
- 경로 효율 (실제/최단)

### 문제 진단
- 가동률 낮음 → 작업 부족 또는 충전 과다
- 효율 낮음 → 경로 문제 또는 병목
- 충돌 많음 → 트래픽 관리 문제""",
        "metadata": {"source": "isaac_sim_docs", "topic": "metrics", "importance": "high"}
    },

    # 고급 설정
    {
        "text": """## 다중 로봇 조율 가이드

### 트래픽 관리 전략
1. **중앙 집중식**
   - 모든 로봇 경로 중앙 서버 관리
   - 최적 경로 계산 가능
   - 단일 장애점 주의

2. **분산식**
   - 로봇 간 직접 통신
   - 로컬 의사결정
   - 확장성 좋음

3. **하이브리드**
   - 전역 계획 + 로컬 회피
   - 실용적 접근

### 우선순위 규칙
- 긴급 작업 우선
- 배터리 부족 로봇 우선 (충전소 접근)
- 적재 로봇 > 공차 로봇
- ID 기반 (동률 시)

### 데드락 방지
- 타임아웃 설정
- 우선순위 역전 감지
- 강제 재경로 계획""",
        "metadata": {"source": "isaac_sim_docs", "topic": "multi_robot", "importance": "medium"}
    },
    {
        "text": """## 배터리 및 충전 전략

### 배터리 모델
- 용량: Ah 또는 Wh
- 방전율: C-rate
- 충전 효율: 85-95%
- 수명: 사이클 수

### 충전 전략
1. **기회 충전**
   - 대기 시간에 짧게 충전
   - 충전소 수 많이 필요
   - 배터리 수명에 영향

2. **완충 충전**
   - 배터리 낮을 때 완충
   - 충전소 수 적음
   - 긴 대기 시간

3. **교체 방식**
   - 배터리 스왑 스테이션
   - 무중단 운영
   - 초기 비용 높음

### 권장 설정
- 충전 임계값: 20-30%
- 복귀 임계값: 15-20%
- 비상 정지: 10% 미만
- 충전소 위치: 작업 영역 외곽""",
        "metadata": {"source": "isaac_sim_docs", "topic": "battery", "importance": "medium"}
    },
    {
        "text": """## 센서 시뮬레이션 가이드

### LiDAR 설정
- 범위: 10-100m
- 해상도: 0.1-1.0도
- 스캔 주기: 10-40 Hz
- 노이즈 모델: 가우시안

### 카메라 설정
- 해상도: 640x480 ~ 4K
- FOV: 60-120도
- 프레임률: 30-60 fps
- 지연: 10-50 ms

### IMU 설정
- 가속도계: ±2g ~ ±16g
- 자이로: ±250 ~ ±2000 dps
- 드리프트: 시간당 누적

### 센서 융합
- 칼만 필터 권장
- 시간 동기화 중요
- 이상치 제거 필요""",
        "metadata": {"source": "isaac_sim_docs", "topic": "sensors", "importance": "medium"}
    },
    {
        "text": """## 성능 최적화 가이드

### GPU 메모리 관리
- 텍스처 품질 조절
- LOD (Level of Detail) 사용
- 불필요한 렌더링 비활성화

### CPU 최적화
- 물리 엔진 스레드 수 조절
- 경로 계획 주기 조절
- 불필요한 로깅 비활성화

### 대규모 시뮬레이션
- 100+ 로봇: 분산 시뮬레이션 고려
- 충돌 감지 최적화 필수
- 시각화 분리 (헤드리스 모드)

### 벤치마크
- 10 로봇: 실시간 × 10 가능
- 50 로봇: 실시간 × 2-5
- 100+ 로봇: 실시간 이하 가능성""",
        "metadata": {"source": "isaac_sim_docs", "topic": "performance", "importance": "medium"}
    }
]

# ============================================================
# 좋은 예시 (Few-shot Learning용)
# ============================================================

GOOD_EXAMPLES = [
    # 기본 창고 예시
    {
        "input": "30m x 20m 창고에 AGV 3대, 선반 2열",
        "output": {
            "environment": {
                "type": "warehouse",
                "width": 30,
                "length": 20,
                "height": 5,
                "floor_friction": 0.7,
                "ambient_temperature": 25,
                "humidity": 50
            },
            "robots": [{
                "type": "AGV",
                "count": 3,
                "speed": 1.5,
                "acceleration": 1.0,
                "payload": 100,
                "battery_capacity": 100,
                "collision_radius": 0.5,
                "sensor_range": 5.0
            }],
            "obstacles": [{
                "type": "shelf",
                "count": 2,
                "arrangement": "row",
                "width": 1.0,
                "length": 2.0,
                "height": 2.0,
                "spacing": 3.0
            }],
            "simulation": {
                "duration": 3600,
                "time_step": 0.01,
                "realtime_factor": 1.0,
                "enable_physics": True,
                "enable_collision": True,
                "record_trajectory": True
            },
            "task": {
                "type": "pickup_delivery",
                "pickup_locations": 5,
                "delivery_locations": 5,
                "task_generation_rate": 0.1
            }
        },
        "explanation": "소형 창고 기본 설정. 로봇 밀도 적정 (200m²당 1대)."
    },
    # 대형 물류센터
    {
        "input": "대형 물류센터 100m x 80m, AGV 10대, 선반 격자 배치, 컨베이어 2개",
        "output": {
            "environment": {
                "type": "warehouse",
                "width": 100,
                "length": 80,
                "height": 12,
                "floor_friction": 0.7,
                "ambient_temperature": 22,
                "humidity": 45
            },
            "robots": [{
                "type": "AGV",
                "count": 10,
                "speed": 1.8,
                "acceleration": 1.2,
                "payload": 200,
                "battery_capacity": 150,
                "collision_radius": 0.6,
                "sensor_range": 8.0
            }],
            "obstacles": [
                {
                    "type": "shelf",
                    "count": 30,
                    "arrangement": "grid",
                    "width": 1.2,
                    "length": 3.0,
                    "height": 3.5,
                    "spacing": 4.0
                },
                {
                    "type": "conveyor",
                    "count": 2,
                    "arrangement": "row",
                    "width": 0.8,
                    "length": 20.0,
                    "height": 0.9,
                    "spacing": 10.0
                }
            ],
            "simulation": {
                "duration": 7200,
                "time_step": 0.01,
                "realtime_factor": 5.0,
                "enable_physics": True,
                "enable_collision": True,
                "record_trajectory": True
            },
            "task": {
                "type": "pickup_delivery",
                "pickup_locations": 20,
                "delivery_locations": 10,
                "task_generation_rate": 0.2
            }
        },
        "explanation": "대형 물류센터. 800m² 당 1대, 충분한 여유. 천장 높이 증가, 배터리 용량 증가."
    },
    # 공장 환경
    {
        "input": "50m x 50m 공장, AMR 5대, 작업대 8개, 컨베이어 3개",
        "output": {
            "environment": {
                "type": "factory",
                "width": 50,
                "length": 50,
                "height": 8,
                "floor_friction": 0.65,
                "ambient_temperature": 28,
                "humidity": 55
            },
            "robots": [{
                "type": "AMR",
                "count": 5,
                "speed": 2.0,
                "acceleration": 1.5,
                "payload": 150,
                "battery_capacity": 120,
                "collision_radius": 0.55,
                "sensor_range": 10.0
            }],
            "obstacles": [
                {
                    "type": "workstation",
                    "count": 8,
                    "arrangement": "grid",
                    "width": 2.0,
                    "length": 2.0,
                    "height": 1.0,
                    "spacing": 5.0
                },
                {
                    "type": "conveyor",
                    "count": 3,
                    "arrangement": "row",
                    "width": 0.6,
                    "length": 15.0,
                    "height": 0.8,
                    "spacing": 8.0
                }
            ],
            "simulation": {
                "duration": 3600,
                "time_step": 0.01,
                "realtime_factor": 1.0,
                "enable_physics": True,
                "enable_collision": True,
                "record_trajectory": True
            },
            "task": {
                "type": "assembly",
                "pickup_locations": 8,
                "delivery_locations": 8,
                "task_generation_rate": 0.15
            }
        },
        "explanation": "공장 환경. AMR 사용 (자율 주행 필요), 공장 온도 높음, 작업대 배치."
    },
    # 소형 연구실
    {
        "input": "작은 연구실 10x10m, 로봇 1대",
        "output": {
            "environment": {
                "type": "custom",
                "width": 10,
                "length": 10,
                "height": 3,
                "floor_friction": 0.6,
                "ambient_temperature": 23,
                "humidity": 50
            },
            "robots": [{
                "type": "AMR",
                "count": 1,
                "speed": 0.8,
                "acceleration": 0.5,
                "payload": 30,
                "battery_capacity": 50,
                "collision_radius": 0.3,
                "sensor_range": 3.0
            }],
            "obstacles": [],
            "simulation": {
                "duration": 1800,
                "time_step": 0.01,
                "realtime_factor": 1.0,
                "enable_physics": True,
                "enable_collision": True,
                "record_trajectory": True
            },
            "task": {
                "type": "patrol",
                "pickup_locations": 3,
                "delivery_locations": 3,
                "task_generation_rate": 0.05
            }
        },
        "explanation": "소형 연구실. 낮은 속도, 작은 적재량, 짧은 시뮬레이션."
    },
    # 고속 분류 센터
    {
        "input": "고속 분류 센터 60x40m, 빠른 AMR 8대, 컨베이어 많이",
        "output": {
            "environment": {
                "type": "warehouse",
                "width": 60,
                "length": 40,
                "height": 6,
                "floor_friction": 0.75,
                "ambient_temperature": 20,
                "humidity": 40
            },
            "robots": [{
                "type": "AMR",
                "count": 8,
                "speed": 3.0,
                "acceleration": 2.0,
                "payload": 50,
                "battery_capacity": 80,
                "collision_radius": 0.4,
                "sensor_range": 12.0
            }],
            "obstacles": [{
                "type": "conveyor",
                "count": 6,
                "arrangement": "row",
                "width": 0.5,
                "length": 30.0,
                "height": 0.7,
                "spacing": 5.0
            }],
            "simulation": {
                "duration": 3600,
                "time_step": 0.005,
                "realtime_factor": 1.0,
                "enable_physics": True,
                "enable_collision": True,
                "record_trajectory": True
            },
            "task": {
                "type": "pickup_delivery",
                "pickup_locations": 15,
                "delivery_locations": 15,
                "task_generation_rate": 0.3
            }
        },
        "explanation": "고속 분류. 빠른 속도(3.0), 높은 가속도, 짧은 time_step(정밀), 많은 컨베이어."
    },
    # 수정 예시들
    {
        "input": "속도 올려줘",
        "context": "기존 설정: AGV 3대, speed: 1.5",
        "output": {"robots": [{"speed": 2.0}]},
        "explanation": "속도 올려 → 20-30% 증가. 1.5 → 2.0"
    },
    {
        "input": "로봇 2대 더 추가",
        "context": "기존 설정: AGV 3대",
        "output": {"robots": [{"count": 5}]},
        "explanation": "2대 더 → 3 + 2 = 5대"
    },
    {
        "input": "시뮬레이션 더 오래 돌려",
        "context": "기존 설정: duration: 3600",
        "output": {"simulation": {"duration": 7200}},
        "explanation": "더 오래 → 2배. 1시간 → 2시간"
    },
    {
        "input": "더 정밀하게",
        "context": "기존 설정: time_step: 0.01",
        "output": {"simulation": {"time_step": 0.005}},
        "explanation": "정밀하게 → time_step 감소. 0.01 → 0.005"
    },
    {
        "input": "로봇 반 줄여",
        "context": "기존 설정: AGV 10대",
        "output": {"robots": [{"count": 5}]},
        "explanation": "반 줄여 → 50% 감소. 10 → 5"
    }
]

# ============================================================
# 분석 예시
# ============================================================

ANALYSIS_EXAMPLES = [
    {
        "config_summary": "30x20m 창고, AGV 3대, 선반 2열",
        "result_summary": "처리량 95/h, 충돌 3회, 병목: intersection_A",
        "analysis": """### 1. 핵심 지표 요약
- **처리량**: 95 items/hour (양호)
- **충돌**: 3회 (개선 필요)
- **병목**: intersection_A (교차로)

### 2. 문제점
- 교차로(intersection_A)에서 충돌 집중
- AGV_1이 2번의 충돌에 관여 → 경로 문제 가능성

### 3. 원인 분석
- 교차로 진입 타이밍 동시 발생
- 우선순위 규칙 부재 또는 미작동

### 4. 개선 권장사항
**즉시 적용:**
- AGV_2 출발 1.5초 지연: `start_delay: 1.5`
- 교차로 속도 제한: `intersection_speed: 0.8`

**중기 개선:**
- 교차로 예약 시스템 도입
- 대체 경로 추가

### 5. Sim2Real 갭 예측
- 예상 현장 처리량: 80-90 items/hour
- 주의: 바닥 상태, 장애물 변동"""
    }
]
