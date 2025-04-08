import os
import time
import json
import numpy as np
from PIL import Image

# GUI 없는 환경용 matplotlib 설정
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

############################################################
# 0) 사전 설정
############################################################

# (A) 경로 관련
model_path = 'model_quant_fixed_edgetpu.tflite'
output_dir = './output_visual'
os.makedirs(output_dir, exist_ok=True)

# (B) TFLite Edge TPU 모델 로드 및 인터프리터 생성
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# (C) 모델 입력 크기 확인
input_width, input_height = common.input_size(interpreter)
print(f"Model input size: {input_width} x {input_height}")

# (D) 색상 팔레트 및 클래스명 정의
#  0: background, 1: dry, 2: humid, 3: slush, 4: snow, 5: wet
palette = np.array([
    [0, 0, 0],         # 0: background
    [113, 193, 255],   # 1: dry
    [255, 219, 158],   # 2: humid
    [125, 255, 238],   # 3: slush
    [235, 235, 235],   # 4: snow
    [255, 61, 61]      # 5: wet
], dtype=np.uint8)

class_names = ["background", "dry", "humid", "slush", "snow", "wet"]

# (E) 레전드(범례) 생성용 패치
legend_patches = []
for i, (r, g, b) in enumerate(palette):
    legend_color = (r/255.0, g/255.0, b/255.0)
    legend_patches.append(mpatches.Patch(color=legend_color, label=class_names[i]))


############################################################
# 1) 세그멘테이션 오버레이 함수
############################################################
def blend_mask(original_np, mask_np, alpha=0.5):
    """
    original_np: (H, W, 3) 원본 이미지 (uint8)
    mask_np:     (H, W)     모델 추론 결과(클래스 인덱스)
    alpha:       마스크의 투명도(0~1)
    """
    overlay = original_np.copy()
    color_mask = palette[mask_np]  # (H, W, 3)
    mask_region = (mask_np != 0)   # 배경(클래스0)은 오버레이 제외
    
    overlay[mask_region] = (
        overlay[mask_region] * (1 - alpha) +
        color_mask[mask_region] * alpha
    ).astype(np.uint8)
    
    return overlay


############################################################
# 2) 메인 루프: 30초 간격으로 카메라 사진 촬영 → 추론 → 시각화·저장
############################################################

interval_seconds = 30  # 30초 간격

print("Starting capture and inference every 30 seconds...")
loop_count = 0

try:
    while True:
        loop_count += 1
        
        # (A) 사진 파일명(타임스탬프)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"capture_{timestamp_str}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        
        # (B) 카메라 촬영(libcamera-still)
        os.system(f"libcamera-still -n -o {img_path} --width 1640 --height 1232")
        
        # (C) 촬영한 이미지 로드
        if not os.path.exists(img_path):
            print(f"[WARNING] Image was not generated: {img_path}")
            time.sleep(interval_seconds)
            continue
        
        print(f"\n[{loop_count}th iteration] New image: {img_path}")
        img_pil = Image.open(img_path).convert('RGB')
        
        # (D) 이미지 전처리(모델 입력 크기에 맞춰 리사이즈)
        resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)
        
        # (E) 모델에 입력 후 추론
        common.set_input(interpreter, resized_img)
        interpreter.invoke()
        mask = segment.get_output(interpreter)
        
        if mask.ndim == 3:
            mask = np.argmax(mask, axis=-1)
        
        # (F) 원본 크기 복원
        orig_width, orig_height = img_pil.size
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        mask_pil = mask_pil.resize((orig_width, orig_height), resample=Image.NEAREST)
        mask_np = np.array(mask_pil)
        
        # (G) 오버레이 이미지 만들기
        orig_np = np.array(img_pil)
        overlay_np = blend_mask(orig_np, mask_np, alpha=0.5)
        
        # (H) 시각화(원본, 세그멘트마스크, 오버레이) 후 저장
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Segmentation Visualization - {img_filename}", fontsize=16)
        
        axes[0].imshow(orig_np)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")
        
        color_mask_np = palette[mask_np]
        axes[1].imshow(color_mask_np)
        axes[1].set_title("Segmentation Mask", fontsize=14)
        axes[1].axis("off")
        
        axes[2].imshow(overlay_np)
        axes[2].set_title("Overlay", fontsize=14)
        axes[2].axis("off")
        
        legend = fig.legend(
            handles=legend_patches,
            loc='center left',
            bbox_to_anchor=(0.92, 0.5),
            fontsize=12
        )
        legend.set_title("Classes", prop={'size': 14})
        legend.get_frame().set_edgecolor("black")
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        out_fig_path = os.path.join(output_dir, f"{os.path.splitext(img_filename)[0]}_visual.png")
        fig.savefig(out_fig_path)
        plt.close(fig)
        
        print(f"Visualization result saved: {out_fig_path}")
        print(f"Waiting {interval_seconds} seconds for next capture...")
        time.sleep(interval_seconds)

except KeyboardInterrupt:
    print("\nCapture and inference loop has been terminated.")
