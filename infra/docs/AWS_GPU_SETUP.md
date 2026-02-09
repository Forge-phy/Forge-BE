# AWS EC2 GPU 인스턴스에서 Forge 실행하기

## 1. EC2 인스턴스 생성

### 추천 인스턴스 타입

| 인스턴스 | GPU | vCPU | RAM | 시간당 비용 | 용도 |
|----------|-----|------|-----|------------|------|
| **g4dn.xlarge** | T4 16GB | 4 | 16GB | ~$0.526 | 개발/테스트 (추천) |
| g4dn.2xlarge | T4 16GB | 8 | 32GB | ~$0.752 | 중간 규모 |
| g5.xlarge | A10G 24GB | 4 | 16GB | ~$1.006 | 고성능 |
| p3.2xlarge | V100 16GB | 8 | 61GB | ~$3.06 | 대규모 시뮬레이션 |

### AMI 선택

**NVIDIA Deep Learning AMI (Ubuntu)** 사용 추천
- AMI 이름: `Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04)`
- 또는: `NVIDIA GPU-Optimized AMI`
- NVIDIA 드라이버, CUDA, Docker가 사전 설치됨

### AWS Console에서 생성

1. EC2 Dashboard → "인스턴스 시작"
2. **이름**: `forge-isaac-sim`
3. **AMI**: "Deep Learning AMI" 검색 → Ubuntu 버전 선택
4. **인스턴스 유형**: `g4dn.xlarge`
5. **키 페어**: 기존 것 선택 또는 새로 생성
6. **네트워크 설정**:
   - 퍼블릭 IP 자동 할당: 활성화
   - 보안 그룹 규칙 추가:
     ```
     SSH       22    내 IP
     HTTP      80    0.0.0.0/0
     Custom    8000  0.0.0.0/0  (Forge API)
     Custom    5173  0.0.0.0/0  (Frontend)
     Custom    8899  0.0.0.0/0  (Isaac Sim Stream)
     ```
7. **스토리지**: 최소 100GB (Isaac Sim 이미지가 큼)
8. 인스턴스 시작

---

## 2. 인스턴스 접속 및 초기 설정

```bash
# SSH 접속
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# GPU 확인
nvidia-smi
```

출력 예시:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   35C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## 3. Docker 및 NVIDIA Container Toolkit 설정

Deep Learning AMI에는 이미 설치되어 있지만, 확인/설치:

```bash
# Docker 확인
docker --version

# NVIDIA Container Toolkit 확인
nvidia-container-cli --version

# 만약 없다면 설치
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 재시작
sudo systemctl restart docker

# GPU Docker 테스트
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## 4. NGC (NVIDIA GPU Cloud) 로그인

Isaac Sim 이미지를 받으려면 NGC 계정이 필요합니다.

```bash
# NGC CLI 설치
wget -O ngc https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod +x ngc-cli/ngc
sudo mv ngc-cli/ngc /usr/local/bin/

# NGC 로그인 (https://ngc.nvidia.com 에서 API Key 발급)
ngc config set

# Docker NGC 로그인
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC_API_KEY>
```

---

## 5. Forge 프로젝트 클론 및 설정

```bash
# 프로젝트 클론
git clone <YOUR_FORGE_REPO_URL> forge
cd forge

# 환경 변수 설정
cat > .env << 'EOF'
ISAAC_SIM_MODE=docker
OPENAI_API_KEY=your_openai_api_key_here
EOF

# Frontend도 클론 (같은 레벨에)
cd ..
git clone <YOUR_FRONTEND_REPO_URL> Frontend
cd forge
```

---

## 6. Isaac Sim 이미지 Pull (시간 소요)

```bash
# Isaac Sim 이미지 다운로드 (약 15-20GB, 시간 소요)
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# 확인
docker images | grep isaac-sim
```

---

## 7. Docker Compose 실행

```bash
# 전체 스택 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 개별 서비스 상태 확인
docker-compose ps
```

예상 출력:
```
NAME                COMMAND                  SERVICE          STATUS
forge-api           "uvicorn api.main:..."   forge-api        running
forge-frontend      "/docker-entrypoint..."  forge-frontend   running
forge-isaac-sim     "./runheadless.nat..."   isaac-sim        running (healthy)
```

---

## 8. 접속 테스트

브라우저에서:
- **Frontend**: `http://<EC2_PUBLIC_IP>:5173`
- **API**: `http://<EC2_PUBLIC_IP>:8000`
- **Isaac Sim Stream**: `http://<EC2_PUBLIC_IP>:8899`

API 상태 확인:
```bash
curl http://localhost:8000/
curl http://localhost:8000/isaac-sim/status
```

---

## 9. 비용 절약 팁

### 인스턴스 중지/시작
```bash
# 작업 끝나면 인스턴스 중지 (Stop) - EBS 비용만 발생
# AWS Console에서 인스턴스 중지

# 다시 시작하면 Public IP가 바뀜
# Elastic IP 할당하면 고정 가능 (월 ~$3.6)
```

### Spot 인스턴스 사용
- g4dn.xlarge Spot: ~$0.16/시간 (70% 절약)
- 중단될 수 있으므로 개발/테스트용으로만

### 자동 종료 스크립트
```bash
# 6시간 후 자동 종료 (비용 방지)
sudo shutdown -h +360
```

---

## 10. 트러블슈팅

### Isaac Sim 컨테이너가 시작되지 않음
```bash
# 로그 확인
docker logs forge-isaac-sim

# GPU 메모리 부족 시
docker-compose down
docker system prune -f
docker-compose up -d
```

### WebRTC 스트림이 안 보임
- 보안 그룹에서 8899 포트 열렸는지 확인
- HTTPS가 필요할 수 있음 (브라우저 정책)

### API 연결 실패
```bash
# Isaac Sim 헬스체크
curl http://localhost:8211/status

# 컨테이너 네트워크 확인
docker network inspect forge_default
```

---

## 빠른 시작 요약

```bash
# 1. EC2 g4dn.xlarge + Deep Learning AMI 생성
# 2. SSH 접속
ssh -i key.pem ubuntu@<IP>

# 3. NGC 로그인
docker login nvcr.io

# 4. 프로젝트 설정
git clone <REPO> forge && cd forge
echo "OPENAI_API_KEY=sk-xxx" > .env

# 5. 실행
docker-compose up -d

# 6. 접속
# http://<IP>:5173
```
