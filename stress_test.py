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

# 모델 로드 및 Edge TPU 인터프리터 초기화
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()
input_width, input_height = common.input_size(interpreter)

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

# 이미지 캡처 및 추론 루프 (스트레스 테스트 용)
while True:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"capture_{timestamp}"
    img_path = os.path.join(capture_dir, f"{img_filename}.jpg")

    capture_cmd = f"libcamera-still -n -o {img_path} --width 1640 --height 1232"
    logging.info(f"Capturing image: {img_filename}")
    os.system(capture_cmd)

    if os.path.exists(img_path):
        start_time = time.time()

        img_pil = Image.open(img_path).convert('RGB').resize((input_width, input_height), resample=Image.LANCZOS)
        common.set_input(interpreter, img_pil)

        interpreter.invoke()
        mask = segment.get_output(interpreter)
        if mask.ndim == 3:
            mask = np.argmax(mask, axis=-1)
        mask = np.array(mask.tolist(), dtype=np.uint8)

        visualize_and_save(img_pil.resize((1640, 1232), Image.LANCZOS), mask, img_filename)

        elapsed_time = time.time() - start_time
        logging.info(f"{img_filename} processed in {elapsed_time:.2f}s")

    else:
        logging.warning(f"Failed to capture {img_filename}")

    logging.info("Waiting for 30 seconds before next capture...")
    time.sleep(30)
