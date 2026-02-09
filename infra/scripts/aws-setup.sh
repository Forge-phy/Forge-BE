#!/bin/bash
# AWS EC2 GPU 인스턴스에서 Forge 빠른 설정 스크립트
# 사용법: curl -sSL <URL> | bash
#
# 전제 조건:
# - g4dn.xlarge 인스턴스 (Deep Learning AMI)
# - 보안 그룹: 22, 80, 5173, 8000, 8899 포트 오픈

set -e

echo "======================================"
echo "  Forge + Isaac Sim 설정 스크립트"
echo "======================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 함수: 성공 메시지
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 함수: 경고 메시지
warn() {
    echo -e "${YELLOW}! $1${NC}"
}

# 함수: 에러 메시지
error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# 1. GPU 확인
echo ""
echo "1. GPU 확인 중..."
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    success "GPU 감지됨: $GPU_NAME"
else
    error "GPU를 찾을 수 없습니다. g4dn 인스턴스인지 확인하세요."
fi

# 2. Docker 확인
echo ""
echo "2. Docker 확인 중..."
if docker --version &> /dev/null; then
    success "Docker 설치됨: $(docker --version)"
else
    warn "Docker 설치 중..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    success "Docker 설치 완료"
fi

# 3. NVIDIA Container Toolkit 확인
echo ""
echo "3. NVIDIA Container Toolkit 확인 중..."
if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    success "NVIDIA Container Toolkit 작동 중"
else
    warn "NVIDIA Container Toolkit 설정 중..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    success "NVIDIA Container Toolkit 설치 완료"
fi

# 4. Docker Compose 확인
echo ""
echo "4. Docker Compose 확인 중..."
if docker compose version &> /dev/null; then
    success "Docker Compose 설치됨"
else
    warn "Docker Compose 설치 중..."
    sudo apt-get install -y docker-compose-plugin
    success "Docker Compose 설치 완료"
fi

# 5. NGC 로그인 확인
echo ""
echo "5. NGC (NVIDIA GPU Cloud) 로그인..."
if docker pull nvcr.io/nvidia/isaac-sim:2023.1.1 --dry-run &> /dev/null 2>&1; then
    success "NGC 이미 로그인됨"
else
    warn "NGC 로그인이 필요합니다."
    echo ""
    echo "NGC API Key를 입력하세요 (https://ngc.nvidia.com 에서 발급):"
    read -sp "NGC API Key: " NGC_KEY
    echo ""

    echo "$NGC_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
    success "NGC 로그인 완료"
fi

# 6. 프로젝트 디렉토리 설정
echo ""
echo "6. 프로젝트 디렉토리 설정..."
WORK_DIR="$HOME/forge-workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Forge 백엔드 클론 (이미 있으면 스킵)
if [ -d "forge" ]; then
    warn "forge 디렉토리가 이미 존재합니다. 업데이트 중..."
    cd forge && git pull && cd ..
else
    echo "Forge 저장소 URL을 입력하세요 (또는 Enter로 스킵):"
    read -p "Git URL: " FORGE_REPO
    if [ -n "$FORGE_REPO" ]; then
        git clone "$FORGE_REPO" forge
    else
        warn "저장소 URL이 없습니다. 수동으로 forge 디렉토리를 생성하세요."
        mkdir -p forge
    fi
fi

# Frontend 클론 (이미 있으면 스킵)
if [ -d "Frontend" ]; then
    warn "Frontend 디렉토리가 이미 존재합니다."
else
    echo "Frontend 저장소 URL을 입력하세요 (또는 Enter로 스킵):"
    read -p "Git URL: " FRONTEND_REPO
    if [ -n "$FRONTEND_REPO" ]; then
        git clone "$FRONTEND_REPO" Frontend
    else
        warn "저장소 URL이 없습니다. 수동으로 Frontend 디렉토리를 생성하세요."
        mkdir -p Frontend
    fi
fi

success "프로젝트 디렉토리: $WORK_DIR"

# 7. 환경 변수 설정
echo ""
echo "7. 환경 변수 설정..."
cd "$WORK_DIR/forge"

if [ ! -f ".env" ]; then
    echo "OpenAI API Key를 입력하세요 (또는 Enter로 스킵):"
    read -sp "OpenAI API Key: " OPENAI_KEY
    echo ""

    cat > .env << EOF
ISAAC_SIM_MODE=docker
OPENAI_API_KEY=${OPENAI_KEY}
EOF
    success ".env 파일 생성됨"
else
    warn ".env 파일이 이미 존재합니다."
fi

# 8. Isaac Sim 이미지 Pull
echo ""
echo "8. Isaac Sim Docker 이미지 다운로드 중..."
echo "   (약 15-20GB, 시간이 걸립니다...)"
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
success "Isaac Sim 이미지 다운로드 완료"

# 9. 실행
echo ""
echo "9. Docker Compose 실행..."
docker compose up -d
success "모든 서비스 시작됨"

# 10. 상태 확인
echo ""
echo "10. 서비스 상태 확인..."
sleep 5
docker compose ps

# 완료 메시지
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_IP")

echo ""
echo "======================================"
echo -e "${GREEN}  설정 완료!${NC}"
echo "======================================"
echo ""
echo "접속 URL:"
echo "  - Frontend:    http://${PUBLIC_IP}:5173"
echo "  - API:         http://${PUBLIC_IP}:8000"
echo "  - API Docs:    http://${PUBLIC_IP}:8000/docs"
echo "  - Isaac Stream: http://${PUBLIC_IP}:8899"
echo ""
echo "유용한 명령어:"
echo "  docker compose logs -f          # 로그 확인"
echo "  docker compose ps               # 상태 확인"
echo "  docker compose down             # 중지"
echo "  docker compose up -d            # 시작"
echo ""
echo -e "${YELLOW}비용 절약: 작업 후 EC2 인스턴스를 중지하세요!${NC}"
echo ""
