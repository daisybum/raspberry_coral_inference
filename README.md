# Raspberry Coral Inference Suite 🦜

라즈베리파이 + Google Coral Edge‑TPU 환경에서  
세그멘테이션 모델 추론·시각화·실시간 카메라 서비스·성능 벤치마크·데이터 EDA·스트레스 테스트까지 **원스톱**으로 수행하는 프로젝트입니다.

---

## ✨ 주요 기능

| 기능 | 설명                                           | 사용 파일                            |
|------|----------------------------------------------|----------------------------------|
| **추론 파이프라인** | Edge‑TPU TFLite 모델 로드 → 전처리 → 추론 → 후처리 → 시각화 | `pipeline.py`, `utils/`          |
| **CLI** | "bench visual camera metric" 4가지 모드 지원       | `inference.py` |
| **벤치마크** | 데이터셋 전체 추론 속도 측정 (ms)                        | `modes/bench_visual.py`          |
| **시각화** | 원본·마스크·오버레이 3‑패널 PNG 저장                      | `modes/bench_visual.py`          |
| **실시간 카메라** | `libcamera-still` 주기 캡처 → 추론·시각화             | `modes/camera.py`                |
| **COCO 메트릭** | PixelAcc / mIoU / Dice / FW‑IoU 계산           | `modes/metric.py`                |
| **스트레스 테스트** | 장시간 추론, CPU 온도·추론 시간 로깅                      | `experiments/stress_test.py`     |
| **센서별 라벨 EDA** | 다중 COCO json 병합 → 센서별 라벨 분포(%, CSV, PNG)     | `experiments/sensor_label_dist.py` |

---

## 🗂 프로젝트 구조
```
├── config.yaml              # 경로·팔레트·클래스 이름 설정
├── inference.py             # 메인 CLI
├── pipeline.py              # SegmentationPipeline
│
├── modes/                   # 실행 모드별 로직
│   ├── bench_visual.py      # 벤치마크 + 시각화
│   ├── camera.py            # 실시간 캡처 추론
│   └── metric.py            # COCO GT 평가
│
├── utils/                   # 공통 헬퍼 모듈
│   ├── image_utils.py
│   ├── file_utils.py
│   ├── visualization.py
│   ├── logger.py
│   └── timer.py
│
└── experiments/             # 특수 실험·EDA
    ├── stress_test.py       # Edge‑TPU 스트레스 테스트
    └── sensor_label_dist.py # 센서별 라벨 분포 EDA
```

## 🐳 설치 (도커 전용)

호스트 OS만 준비되면 Docker 이미지 안에 Python, PyCoral, 필수 라이브러리가 모두 포함되어 있습니다.  
NVIDIA GPU가 없는 Raspberry Pi 환경에서도 그대로 동작합니다.


| 단계             | 명령                                                            | 설명                                             |
|------------------|---------------------------------------------------------------|--------------------------------------------------|
| 1️⃣ 이미지 빌드   | ```docker compose -f docker/docker-compose.yml build```       | Dockerfile → coral-inference 이미지 생성         |
| 2️⃣ 컨테이너 시작 | ```docker compose -f docker/docker-compose.yml up -d``` | 백그라운드(detached) 모드                        |
| 3️⃣ 쉘 진입       | ```docker compose exec -it coral-inference /bin/bash``` | 컨테이너 안에서 `python inference.py ...` 실행  |


## 🚀 CLI 사용법

### 벤치마크 (평균 추론 시간)
python inference.py --mode bench

### 데이터셋 시각화 PNG 저장 (config.paths.output_dir)
python inference.py --mode visual

### 실시간 카메라 추론 (15초 간격)
python inference.py --mode camera --interval 15

### COCO GT 평가 (mIoU·Dice·PixelAcc·FW‑IoU)
python inference.py --mode metric


