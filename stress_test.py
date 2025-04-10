import os
import time
import logging
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 기본 설정
model_path = 'model_quant_fixed_edgetpu.tflite'
capture_dir = '/media/pi/ESD-USB/captured_images'
output_dir = '/media/pi/ESD-USB/output_visual'
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# CPU 온도 측정 함수
def get_cpu_temperature():
    try:
        temp_str = os.popen("vcgencmd measure_temp").readline()
        # 예: temp=45.0'C 형태의 문자열 파싱
        temp_value = float(temp_str.replace("temp=", "").replace("'C", "").strip())
        return temp_value
    except Exception as e:
        logging.error("CPU 온도 측정 오류: " + str(e))
        return None

# 모델 로드 및 Edge TPU 인터프리터 초기화
try:
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_width, input_height = common.input_size(interpreter)
except Exception as e:
    logging.error("모델 로드 또는 인터프리터 초기화 오류: " + str(e))
    raise

# 컬러 팔레트
palette = np.array([
    [0, 0, 0],
    [113, 193, 255],
    [255, 219, 158],
    [125, 255, 238],
    [235, 235, 235],
    [255, 61, 61]
], dtype=np.uint8)

# 시각화 및 저장 함수
def visualize_and_save(img_pil, mask, img_filename):
    try:
        orig_np = np.array(img_pil)
        mask_np = np.array(Image.fromarray(mask.astype(np.uint8)).resize(img_pil.size, resample=Image.NEAREST))
        color_mask_np = palette[mask_np]

        overlay = orig_np.copy()
        alpha = 0.5
        mask_region = (mask_np != 0)
        overlay[mask_region] = (overlay[mask_region] * (1 - alpha) + color_mask_np[mask_region] * alpha).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(color_mask_np)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        visual_path = os.path.join(output_dir, f'{img_filename}_visual.png')
        plt.savefig(visual_path)
        plt.close()
        logging.info(f"Visualization saved: {visual_path}")
    except Exception as e:
        logging.error("Visualization 및 저장 오류: " + str(e))

# 이미지 캡처 및 추론 루프 (스트레스 테스트용)
while True:
    try:
        # CPU 온도 로깅
        cpu_temp = get_cpu_temperature()
        if cpu_temp is not None:
            logging.info(f"현재 CPU 온도: {cpu_temp:.2f}°C")
        else:
            logging.warning("CPU 온도를 측정할 수 없습니다.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"capture_{timestamp}"
        img_path = os.path.join(capture_dir, f"{img_filename}.jpg")

        capture_cmd = f"libcamera-still -n -o {img_path} --width 1640 --height 1232"
        logging.info(f"이미지 캡처 시작: {img_filename}")
        os.system(capture_cmd)

        if os.path.exists(img_path):
            start_time = time.time()
            
            # 이미지 로딩 및 리사이즈
            try:
                img_pil = Image.open(img_path).convert('RGB').resize((input_width, input_height), resample=Image.LANCZOS)
            except Exception as e:
                logging.error("이미지 로드 오류: " + str(e))
                continue

            common.set_input(interpreter, img_pil)

            try:
                interpreter.invoke()
            except Exception as e:
                logging.error("모델 추론 호출 오류: " + str(e))
                continue

            try:
                mask = segment.get_output(interpreter)
                if mask.ndim == 3:
                    mask = np.argmax(mask, axis=-1)
                mask = np.array(mask.tolist(), dtype=np.uint8)
            except Exception as e:
                logging.error("추론 결과 처리 오류: " + str(e))
                continue

            visualize_and_save(img_pil.resize((1640, 1232), Image.LANCZOS), mask, img_filename)

            elapsed_time = time.time() - start_time
            logging.info(f"{img_filename} 처리 완료 - 소요 시간: {elapsed_time:.2f}s")
        else:
            logging.warning(f"{img_filename} 캡처 실패")

    except Exception as e:
        logging.error("시스템 오류 발생: " + str(e))

    logging.info("다음 캡처 전 30초 대기...")
    time.sleep(30)
