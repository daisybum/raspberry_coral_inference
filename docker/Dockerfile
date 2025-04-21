# ──────────────────────────────────────────────────────────────
# Raspberry Pi Bullseye 64‑bit · Python 3.8 슬림 베이스
# ──────────────────────────────────────────────────────────────
FROM arm64v8/python:3.8-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# 0) Release‑file 유효기간 검사 비활성화(시계 오차 방지)
RUN echo 'Acquire::Check-Valid-Until "false";' \
        > /etc/apt/apt.conf.d/99no-check-valid-until

# 1) 기본 툴 + 저장소 등록
RUN set -eux; \
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg apt-transport-https build-essential

# ── Coral Edge TPU 저장소 / 키 ─────────────────────────────
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor -o /usr/share/keyrings/coral.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral.gpg] \
         https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list

# ── Raspberry Pi 저장소 / 키 (picamera2 · libcamera‑apps) ─
RUN curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | \
        gpg --dearmor -o /usr/share/keyrings/raspi.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/raspi.gpg] \
         http://archive.raspberrypi.org/debian bullseye main" \
        > /etc/apt/sources.list.d/raspi.list

# 2) Edge TPU 런타임 + 카메라 앱 등 설치
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
        libedgetpu1-std \
        python3-tflite-runtime \
        python3-pip \
        python3-picamera2 \
        libcamera-apps \            # libcamera‑still 포함
        libhdf5-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# 3) python → python3 심볼릭 링크
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --set python /usr/bin/python3

# 4) 파이썬 패키지
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ \
        pycoral~=2.0 && \
    pip install --no-cache-dir \
        tqdm pillow pycocotools pyyaml tensorflow

# 5) 작업 디렉터리
WORKDIR /app
