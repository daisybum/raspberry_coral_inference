import os
import time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 백엔드 설정
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# 0) 경로 및 기본 설정
model_path = 'model_quant_fixed_edgetpu.tflite'
capture_dir = './captured_images'
output_dir = './output_visual'
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 1) TFLite 모델 로드 및 Edge TPU interpreter 초기화
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()
input_width, input_height = common.input_size(interpreter)
print(f"모델 입력 크기: {input_width} x {input_height}")

# 2) 색상 팔레트 및 클래스 이름 (0: 배경, 1: dry, 2: humid, 3: slush, 4: snow, 5: wet)
palette = np.array([
    [0, 0, 0],         # 0: 배경
    [113, 193, 255],   # 1: dry
    [255, 219, 158],   # 2: humid
    [125, 255, 238],   # 3: slush
    [235, 235, 235],   # 4: snow
    [255, 61, 61]      # 5: wet
], dtype=np.uint8)
class_names = ["background", "dry", "humid", "slush", "snow", "wet"]

# 3) blending 함수를 정의 (원본 이미지와 segmentation 마스크를 합성)
def blend_mask(original_np, mask_np, alpha=0.5):
    """
    original_np: (H, W, 3) 원본 이미지 배열 (uint8)
    mask_np:     (H, W) 클래스 인덱스 배열 (0~5)
    alpha:       마스크 투명도 (0~1)
    """
    overlay = original_np.copy()
    color_mask = palette[mask_np]  # (H, W, 3)
    mask_region = (mask_np != 0)   # 배경이 아닌 부분에 대해서만 합성
    overlay[mask_region] = (
        overlay[mask_region] * (1 - alpha) +
        color_mask[mask_region] * alpha
    ).astype(np.uint8)
    return overlay

# 4) matplotlib 범례 생성: 각 클래스별 색상과 이름 설정
legend_patches = []
for i, (r, g, b) in enumerate(palette):
    legend_color = (r/255, g/255, b/255)  # matplotlib는 0~1 범위 사용
    legend_patches.append(mpatches.Patch(color=legend_color, label=class_names[i]))

# 5) 이미지 처리 및 시각화를 위한 함수 정의
def process_image(img_path):
    start_time = time.time()
    # 이미지 로드 및 원본 크기 얻기
    img_pil = Image.open(img_path).convert('RGB')
    orig_width, orig_height = img_pil.size
    print(f"처리할 이미지: {img_path} (원본 크기: {orig_width}x{orig_height})")
    
    # (1) 모델 입력 크기로 이미지 리사이즈 (LANCZOS 필터 사용)
    resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)
    
    # (2) 추론: 리사이즈한 이미지를 모델 입력으로 설정 후 추론 수행
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    mask = segment.get_output(interpreter)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    
    # (3) 추론 결과 마스크를 원본 크기로 리사이즈 (최근접 보간법 사용)
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    mask_pil = mask_pil.resize((orig_width, orig_height), resample=Image.NEAREST)
    mask_np = np.array(mask_pil)
    
    # (4) 색상 마스크 생성: 각 픽셀에 대해 클래스 색상 적용
    color_mask_np = palette[mask_np]
    
    # (5) 원본 이미지와 마스크를 합성하여 오버레이 이미지 생성
    orig_np = np.array(img_pil.resize((orig_width, orig_height), resample=Image.LANCZOS))
    overlay_np = blend_mask(orig_np, mask_np, alpha=0.5)
    
    # (6) Matplotlib을 활용해 3 부분(원본, 마스크, 오버레이)으로 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"세그멘테이션 시각화 - {os.path.basename(img_path)}", fontsize=16)
    
    # 원본 이미지 서브플롯
    axes[0].imshow(orig_np)
    axes[0].set_title("원본 이미지", fontsize=14)
    axes[0].axis("off")
    
    # 세그멘테이션 마스크 서브플롯
    axes[1].imshow(color_mask_np)
    axes[1].set_title("세그멘테이션 마스크", fontsize=14)
    axes[1].axis("off")
    
    # 오버레이 이미지 서브플롯
    axes[2].imshow(overlay_np)
    axes[2].set_title("오버레이", fontsize=14)
    axes[2].axis("off")
    
    # 범례 추가 (오른쪽에 배치)
    legend = fig.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),
        fontsize=12,
        title_fontsize=14
    )
    legend.get_frame().set_edgecolor("black")
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # 결과 저장: 캡처 이미지의 파일명에 _visual 추가하여 저장
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_fig_path = os.path.join(output_dir, f"{base_name}_visual.png")
    fig.savefig(out_fig_path)
    plt.close(fig)
    
    elapsed = time.time() - start_time
    print(f"시각화 저장: {out_fig_path}")
    print(f"이미지 처리 시간: {elapsed:.2f}초")
    print("-" * 50)

# 6) 30초 간격으로 이미지 캡처 및 처리하는 메인 루프
while True:
    # 현재 시간 기반의 고유 파일명 생성 (예: capture_20250408_153000.jpg)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"capture_{timestamp}.jpg"
    img_path = os.path.join(capture_dir, img_filename)
    
    # libcamera를 사용해 이미지 캡처 (GUI 없이 캡처)
    capture_cmd = f"libcamera-still -n -o {img_path} --width 1640 --height 1232"
    print(f"[정보] 이미지 캡처 시작: {capture_cmd}")
    os.system(capture_cmd)
    
    # 캡처된 이미지가 존재하면 추론 및 시각화 실행
    if os.path.exists(img_path):
        process_image(img_path)
    else:
        print(f"[경고] 이미지 캡처 실패: {img_path} 파일이 존재하지 않음")
    
    # 30초 대기 후 다음 캡처 실행
    print("30초 대기 중...\n")
    time.sleep(30)
