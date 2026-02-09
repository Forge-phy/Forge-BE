# Forge Isaac Sim - Custom Image
FROM nvcr.io/nvidia/isaac-sim:2023.1.1

# Extension 복사
COPY omni.forge.api /isaac-sim/exts/omni.forge.api

# 수정된 kit 파일 복사
COPY omni.isaac.sim.forge.kit /isaac-sim/apps/omni.isaac.sim.forge.kit

# 시작 스크립트 복사
COPY start.sh /isaac-sim/start.sh
RUN chmod +x /isaac-sim/start.sh

# 포트 노출
EXPOSE 9999

# ENTRYPOINT 제거하고 직접 실행
ENTRYPOINT []
CMD ["/bin/bash", "/isaac-sim/start.sh"]
