# Raspberry Pi Bullseye 64‑bit Python 3.8 슬림 베이스 이미지
FROM arm64v8/python:3.8-slim-bullseye

# 비대화식 모드 설정
ENV DEBIAN_FRONTEND=noninteractive

# 1) APT 저장소 등록 및 필수 패키지 설치
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        build-essential; \
    \
    # Coral EdgeTPU 저장소
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | tee /etc/apt/sources.list.d/coral-edgetpu.list; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -; \
    \
    # Raspberry Pi 전용 저장소 (picamera2 등)
    echo "deb http://archive.raspberrypi.org/debian bullseye main" \
        | tee /etc/apt/sources.list.d/raspi.list; \
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add -; \
    \
    # Edge TPU 런타임 및 기타 필수 패키지 설치
    apt-get update && apt-get install -y --no-install-recommends \
        libedgetpu1-std \
        python3-tflite-runtime \
        python3-pip \
        pkg-config \
        libhdf5-dev \
        python3-picamera2; \
    \
    # APT 캐시 정리
    rm -rf /var/lib/apt/lists/*

# 2) python → python3 심볼릭 링크
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --set python /usr/bin/python3

# 3) pip 최신화 및 파이썬 패키지 설치
RUN python -m pip install --upgrade pip setuptools wheel && \
    \
    # Coral PyPI 저장소를 통해 pycoral 2.x 설치
    python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0 && \
    \
    # 기타 공통 라이브러리
    pip install --no-cache-dir \
        tqdm \
        pillow \
        pycocotools \
        pyyaml \
        tensorflow

# 4) 작업 디렉터리
WORKDIR /app
