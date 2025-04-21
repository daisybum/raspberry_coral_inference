# Raspberry Pi Bullseye 64bit 환경용 Python 3.8 슬림 베이스 이미지
FROM arm64v8/python:3.8-slim-bullseye

# 1) 비대화식 모드 설정 (APT 설치 시 대화창 방지)
ENV DEBIAN_FRONTEND=noninteractive

# 2) 필수 패키지 설치 및 저장소 등록
RUN set -eux; \
    # 기본 패키지 설치 (APT 업데이트 후 필요한 도구들 설치)
    apt-get update && apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        build-essential; \
    \
    # Coral EdgeTPU 관련 패키지 설치를 위한 저장소 등록 및 GPG 키 추가
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
         | tee /etc/apt/sources.list.d/coral-edgetpu.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -; \
    \
    # Raspberry Pi 전용 패키지 제공을 위한 저장소 추가 (예: picamera2)
    echo "deb http://archive.raspberrypi.org/debian bullseye main" \
         | tee /etc/apt/sources.list.d/raspi.list; \
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | apt-key add -; \
    \
    # libcamera 빌드에 필요한 의존성 패키지 설치
    apt-get update && apt-get install -y --no-install-recommends \
        libboost-dev \
        libgnutls28-dev \
        openssl \
        libtiff-dev \
        pybind11-dev \
        qtbase5-dev \
        libqt5core5a \
        libqt5widgets \
        meson \
        cmake \
        python3-yaml \
        python3-ply \
        libglib2.0-dev \
        libgstreamer-plugins-base1.0-dev \
        git; \
    \
    # Coral EdgeTPU 및 기타 필수 패키지 설치
    apt-get install -y --no-install-recommends \
        libedgetpu1-std \
        python3-pycoral \
        python3-tflite-runtime \
        python3-pip \
        pkg-config \
        libhdf5-dev \
        python3-picamera2; \
    \
    # APT 캐시 정리
    rm -rf /var/lib/apt/lists/*

# 3) libcamera 소스 클론 및 빌드
RUN git clone https://github.com/raspberrypi/libcamera.git /libcamera \
    && cd /libcamera \
    && meson setup build --buildtype=release -Dpipelines=rpi/vc4,rpi/pisp -Dipas=rpi/vc4,rpi/pisp -Dv4l2=true -Dgstreamer=enabled -Dtest=false -Dlc-compliance=disabled -Dcam=disabled -Dqcam=disabled -Ddocumentation=disabled -Dpycamera=enabled \
    && ninja -C build install

# 4) python 명령어를 python3와 동일하게 연결
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --set python /usr/bin/python3

# 5) pip 최신화 및 추가 파이썬 라이브러리 설치
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        tqdm \
        pillow \
        pycocotools \
        pyyaml \
        tensorflow

# 6) 작업 디렉터리 설정
WORKDIR /app
