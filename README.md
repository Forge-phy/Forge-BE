# Forge Backend

**Forge** — 자연어 Isaac Sim 시뮬레이션 + Sim2Real 갭 예측 엔진

Talos Physical AI Platform의 핵심 솔루션으로, 자연어 입력만으로 Isaac Sim 시뮬레이션을 생성하고 Sim2Real 갭을 예측합니다.

## 주요 기능

- **자연어 → 시뮬레이션**: 자연어 입력을 Isaac Sim 시뮬레이션 파라미터로 변환
- **LLM 하이브리드**: 민감도 기반 OpenAI / Qwen 자동 분기
- **RAG 기반 검색**: ChromaDB 벡터 검색으로 유사 시뮬레이션 참조
- **Sim2Real 갭 예측**: LSTM 기반 시뮬레이션-실제 성능 갭 예측
- **파이프라인 자동화**: NL → Isaac Sim → LSTM → 보고서 생성

## 기술 스택

- **API Server**: FastAPI (Python 3.11)
- **LLM**: OpenAI GPT + Ollama (Qwen)
- **Vector DB**: ChromaDB + sentence-transformers
- **ML**: PyTorch LSTM
- **Simulator**: NVIDIA Isaac Sim 2023.1.1
- **Infra**: Docker Compose, WebRTC Streaming

## 프로젝트 구조

```
backend/
├── api/          # FastAPI 메인 라우터
├── llm/          # LLM 통합 모듈 (OpenAI/Ollama)
├── rag/          # RAG 검색 증강 생성
├── isaac/        # Isaac Sim 연동
├── lstm/         # Sim2Real 갭 예측 (LSTM)
└── tests/        # 테스트 코드

infra/
├── docker-compose.yml
├── Dockerfile
├── isaac-docker/     # Isaac Sim Docker 확장
├── isaac-extension/  # Isaac Sim 로컬 확장
├── docs/             # 문서
└── scripts/          # 스크립트
```

## 실행 방법

### Docker Compose (권장)

```bash
cd infra
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 설정
docker compose up -d
```

### 로컬 실행

```bash
cd backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/llm/environment` | POST | 자연어 → Isaac Sim 파라미터 |
| `/llm/modify` | POST | 대화형 파라미터 수정 |
| `/llm/analyze` | POST | 시뮬레이션 결과 분석 |
| `/isaac-sim/execute` | POST | Isaac Sim 명령 실행 |
| `/lstm/predict` | POST | Sim2Real 갭 예측 |
| `/pipeline/run` | POST | 전체 파이프라인 실행 |

## 환경변수

| 변수 | 설명 | 필수 |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 | O |
| `OPENAI_MODEL` | OpenAI 모델명 | X (기본: gpt-4o-mini) |
| `OLLAMA_MODEL` | Ollama 모델명 | X (기본: qwen2.5:7b) |
| `ISAAC_SIM_MODE` | Isaac Sim 모드 (mock/docker/local/aws) | X (기본: mock) |

## 라이선스

MIT License
