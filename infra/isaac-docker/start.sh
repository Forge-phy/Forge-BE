#!/bin/bash
# Forge Isaac Sim 시작 스크립트

cd /isaac-sim

# kit 직접 실행 - 우리 Extension 포함
./kit/kit \
    ./apps/omni.isaac.sim.forge.kit \
    --ext-folder ./exts \
    --enable omni.forge.api \
    --no-window \
    --allow-root
