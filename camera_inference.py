import os
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # GUI 디스플레이 없이 Matplotlib 사용 (Agg 백엔드)

# Picamera2 및 PyCoral 관련 모듈 임포트
from picamera2 import Picamera2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, segment

# 1) 결과 디렉토리 준비
os.makedirs("/workspace/captured_images", exist_ok=True)
os.makedirs("/workspace/output_visual", exist_ok=True)

# 2) 카메라 초기화 및 설정
picam2 = Picamera2()
camera_config = picam2.create_still_configuration()  # 기본 스틸 캡처 설정 (해상도 기본값 사용)
picam2.configure(camera_config)
picam2.start(show_preview=False)  # 헤드리스 모드: 프리뷰 비활성화
time.sleep(2)  # 카메라 센서 안정화 대기 (노출 등 조정)

# 3) PyCoral 세그멘테이션 모델 로드
MODEL_PATH = "/workspace/model_deeplabv3.tflite"  # 사용자가 준비한 TFLite 세그멘테이션 모델 경로
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
model_width, model_height = common.input_size(interpreter)  # 모델 입력 해상도 획득

# (선택) 세그멘테이션 결과를 색상화하기 위한 컬러맵 함수 정의
def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3
    return colormap

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")
    return colormap[label]

# 4) 주기적 캡처 및 추론 루프
try:
    while True:
        # (a) 이미지 캡처
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_path = f"/workspace/captured_images/img_{timestamp}.jpg"
        picam2.capture_file(img_path)  # 사진 촬영 및 저장&#8203;:contentReference[oaicite:3]{index=3}

        # (b) 추론을 위해 이미지 로드 및 전처리
        image = Image.open(img_path).convert("RGB")
        # 모델 입력 크기로 이미지 리사이즈 (필요 시 모델 비율에 맞게 조정)
        resized_image = image.resize((model_width, model_height), Image.ANTIALIAS)

        # (c) PyCoral Edge TPU를 이용한 세그멘테이션 추론
        common.set_input(interpreter, resized_image)    # 입력 텐서 설정&#8203;:contentReference[oaicite:4]{index=4}
        interpreter.invoke()                            # 추론 실행&#8203;:contentReference[oaicite:5]{index=5}
        seg_map = segment.get_output(interpreter)       # 세그멘테이션 출력 획득&#8203;:contentReference[oaicite:6]{index=6}
        if seg_map.ndim == 3:                           # 모델 출력이 (H,W,C)일 경우 다중채널 확률맵으로 간주
            seg_map = np.argmax(seg_map, axis=-1)       # 채널 축에 argmax를 적용하여 클래스 맵으로 변환

        # (d) 세그멘테이션 결과를 컬러 마스크 이미지로 변환
        color_mask = label_to_color_image(seg_map).astype(np.uint8)  # 클래스 맵을 색상 이미지로&#8203;:contentReference[oaicite:7]{index=7}
        mask_img = Image.fromarray(color_mask)

        # (e) 원본 이미지와 마스크를 결합하여 출력 이미지 생성
        output_img = Image.new("RGB", (model_width * 2, model_height))
        output_img.paste(resized_image, (0, 0))               # 좌측에 원본(리사이즈된) 이미지 붙여넣기
        output_img.paste(mask_img, (model_width, 0))          # 우측에 마스크 이미지 붙여넣기&#8203;:contentReference[oaicite:8]{index=8}

        # (f) 결과 이미지 파일로 저장
        out_path = f"/workspace/output_visual/result_{timestamp}.jpg"
        output_img.save(out_path)  # 최종 이미지 저장&#8203;:contentReference[oaicite:9]{index=9}

        # (g) 30초 대기 후 다음 루프
        time.sleep(30)  # 30초마다 주기적으로 실행&#8203;:contentReference[oaicite:10]{index=10}
except KeyboardInterrupt:
    picam2.stop()  # 스크립트 종료 시 카메라 정지
