# RunPod GPU Pod에서 Forge 실행하기

## 1. RunPod Pod 생성

### 추천 GPU

| GPU | VRAM | RT Core | 시간당 비용 | 용도 |
|-----|------|---------|------------|------|
| **RTX A6000** | 48GB | ✅ | ~$0.33 | 개발/테스트 (추천) |
| L40S | 48GB | ✅ | ~$0.79 | 고성능 |
| RTX 6000 Ada | 48GB | ✅ | ~$0.89 | 최신 아키텍처 |

> Isaac Sim은 **RT Core GPU 필수** (A100/H100 미지원)

### CLI로 생성

```bash
runpodctl pod create \
  --name "forge-dev" \
  --gpu-id "NVIDIA RTX A6000" \
  --gpu-count 1 \
  --image "nvidia/cuda:12.2.0-devel-ubuntu22.04" \
  --container-disk-in-gb 50 \
  --volume-in-gb 100 \
  --ports "8888/http,8000/http,5173/http,8899/http,22/tcp"
```

---

## 2. Pod 접속

```bash
# SSH 접속
ssh root@<POD_IP> -p <PORT> -i ~/.runpod/ssh/RunPod-Key-Go

# 또는 RunPod 웹 터미널 사용

# GPU 확인
nvidia-smi
```

---

## 3. 자동 설정 (권장)

```bash
# 설정 스크립트 실행
cd /workspace
git clone <YOUR_FORGE_REPO_URL> forge
cd forge/infra/scripts
bash runpod-startup.sh
```

---

## 4. 수동 설정

```bash
# 프로젝트 클론
cd /workspace
git clone <YOUR_FORGE_REPO_URL> forge
git clone <YOUR_FRONTEND_REPO_URL> frontend

# 환경 변수 설정
cd forge
cat > infra/.env << 'EOF'
ISAAC_SIM_MODE=docker
OPENAI_API_KEY=your_openai_api_key_here
EOF

# NGC 로그인 (Isaac Sim 이미지 다운로드에 필요)
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC_API_KEY>

# Isaac Sim 이미지 다운로드 (~15-20GB)
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# 실행
cd infra
docker compose up -d
```

---

## 5. 접속 테스트

RunPod Proxy를 통해 접속:
- **Frontend**: `https://<POD_ID>-5173.proxy.runpod.net`
- **API**: `https://<POD_ID>-8000.proxy.runpod.net`
- **API Docs**: `https://<POD_ID>-8000.proxy.runpod.net/docs`
- **Isaac Sim Stream**: `https://<POD_ID>-8899.proxy.runpod.net`

```bash
curl http://localhost:8000/
curl http://localhost:8000/isaac-sim/status
```

---

## 6. 비용 절약

### Pod Stop/Start

```bash
# CLI로 Pod 중지 (GPU 과금 중단, 스토리지만 소액 과금)
runpodctl pod stop <POD_ID>

# Pod 재시작
runpodctl pod start <POD_ID>
```

### 월 비용 추정

| 사용 패턴 | 비용 |
|-----------|------|
| 4hr/일 × 20일 | ~$26 (≈ ₩35,000) |
| 2hr/일 × 15일 | ~$10 (≈ ₩13,000) |
| 24hr × 30일 (비추천) | ~$238 (≈ ₩320,000) |

---

## 7. 트러블슈팅

### Isaac Sim 컨테이너가 시작되지 않음
```bash
docker logs forge-isaac-sim
docker compose down && docker compose up -d
```

### GPU 메모리 부족
```bash
# Ollama 모델이 GPU 메모리를 점유 중일 수 있음
docker restart forge-ollama
```

### WebRTC 스트림이 안 보임
- RunPod Proxy URL로 접속하고 있는지 확인
- HTTPS 필요 (브라우저 정책)

---

## 빠른 시작 요약

```bash
# 1. RunPod RTX A6000 Pod 생성
# 2. SSH 또는 웹 터미널 접속
# 3. 설정 스크립트 실행
cd /workspace
git clone <REPO> forge && cd forge/infra/scripts
bash runpod-startup.sh

# 4. 접속: https://<POD_ID>-5173.proxy.runpod.net
# 5. 끝나면 Pod Stop!
```
