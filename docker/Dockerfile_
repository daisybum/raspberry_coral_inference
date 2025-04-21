# ──────────────────────────────────────────────────────────────
# Raspberry Pi Bullseye 64‑bit · Python 3.9 Slim (ARM64)
# ──────────────────────────────────────────────────────────────
FROM --platform=linux/arm64/v8 arm64v8/python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=2000 \
    PIP_RETRIES=5

# 1) 핵심 빌드 도구 및 의존성 + Coral/RPi 저장소 추가
RUN set -eux; \
    # ─ apt 기본 툴 ─
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg apt-transport-https build-essential \
        libhdf5-dev pkg-config python3-dev cython3 && \
    \
    # ─ Coral EdgeTPU repo & key ─
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list && \
    \
    # ─ Raspberry Pi repo & key (picamera2 등) ─
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | gpg --dearmor -o /usr/share/keyrings/raspi.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspi.gpg] http://archive.raspberrypi.org/debian bullseye main" \
        > /etc/apt/sources.list.d/raspi.list && \
    \
    # ─ EdgeTPU·카메라 런타임 ─
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
        libedgetpu1-std python3-tflite-runtime python3-picamera2 libcamera-apps && \
    \
    # ─ 캐시 정리 ─
    rm -rf /var/lib/apt/lists/*

# 2) Python 패키지 (휠 사용해 빌드 시간 최소화)
# piwheels 우선, 없으면 PyPI로 fallback
RUN python -m pip install --upgrade pip setuptools wheel && \
    # Coral 전용
    python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ \
        pycoral~=2.0 && \
    # 공통 휠 (piwheels 우선)
    pip install --prefer-binary \
        --extra-index-url https://www.piwheels.org/simple \
        pillow tqdm pycocotools==2.0.8 numpy==1.26.3 tensorflow && \
    # 남은 휠 캐시 제거
    rm -rf /root/.cache/pip

# 3) 작업 디렉터리
WORKDIR /app
