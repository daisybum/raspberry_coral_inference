import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# 0) 경로 설정
annotations_path = '/media/shpark/new-volumn/merged_all/test_annotations.coco.json'
image_dir = '/media/shpark/new-volumn/merged_all'
model_path = 'model_quant_fixed_edgetpu.tflite'

# 세그멘테이션 결과 시각화 이미지를 저장할 폴더 (필요하다면 사용)
output_dir = 'output_visual'
os.makedirs(output_dir, exist_ok=True)

# 1) Edge TPU용 TFLite 모델 로드 및 인터프리터 초기화
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# 2) 모델 입력 크기
input_width, input_height = common.input_size(interpreter)
print(f"모델 입력 크기: {input_width} x {input_height}")

# 3) COCO 형식 JSON 파일 로드
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# 4) 클래스 인덱스별 색상 팔레트와 클래스 이름 정의
#  0: background, 1: dry, 2: humid, 3: slush, 4: snow, 5: wet
palette = np.array([
    [0, 0, 0],    
    [128, 255, 0],    
    [0, 128, 255],    
    [255, 0, 128],   
    [128, 0, 255],
    [255, 255, 0]    
], dtype=np.uint8)

class_names = ["background", "dry", "humid", "slush", "snow", "wet"]

# 5) 오버레이를 위한 함수
def blend_mask(original_np, mask_np, alpha=0.5):
    """
    original_np: (H, W, 3) uint8 원본 이미지 배열
    mask_np:     (H, W)     픽셀별 클래스 인덱스 (0~5)
    alpha:       마스크 투명도 (0~1)
    """
    overlay = original_np.copy()
    color_mask = palette[mask_np]  # (H, W, 3)
    # 원하는 규칙에 따라 mask_np != 0 부분만 덧씌우거나, 전체 픽셀에 덧씌울 수도 있음
    # 여기서는 mask_np != 0 (0번을 배경으로 가정) 영역만 오버레이
    mask_region = (mask_np != 0)
    overlay[mask_region] = (
        overlay[mask_region] * (1 - alpha) +
        color_mask[mask_region] * alpha
    ).astype(np.uint8)
    return overlay

# 6) 범례(legend) 패치 생성 (각 패치에 라벨 이름 포함)
legend_patches = []
for i, (r, g, b) in enumerate(palette):
    legend_color = (r/255, g/255, b/255)  # matplotlib는 0~1 범위의 색상을 사용
    # mpatches.Patch에 label 매개변수를 사용해 클래스 이름을 지정
    legend_patches.append(mpatches.Patch(color=legend_color, label=class_names[i]))

# 7) 이미지 순회하며 추론 및 시각화
for image_info in coco_data['images']:
    file_name = image_info['file_name']
    orig_width = image_info['width']
    orig_height = image_info['height']

    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        print(f"[WARN] 이미지 파일이 없음: {img_path}")
        continue

    print(f"처리 중: {img_path}")

    # (1) 원본 이미지 로드 및 리사이즈
    img_pil = Image.open(img_path).convert('RGB')
    resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)

    # (2) 세그멘테이션 추론
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    mask = segment.get_output(interpreter)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)

    # (3) 원본 크기로 마스크 리사이즈
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    mask_pil = mask_pil.resize((orig_width, orig_height), resample=Image.NEAREST)
    mask_np = np.array(mask_pil)

    # (4) 컬러 마스크 생성
    color_mask_np = palette[mask_np]

    # (5) 오버레이 이미지 생성
    orig_np = np.array(img_pil)
    if (orig_np.shape[1] != orig_width) or (orig_np.shape[0] != orig_height):
        orig_np = np.array(img_pil.resize((orig_width, orig_height), resample=Image.LANCZOS))
    overlay_np = blend_mask(orig_np, mask_np, alpha=0.5)

    # (6) Matplotlib 시각화 (원본, 마스크, 오버레이 3분할)
    # 전체 그림 크기를 넓게 잡아 범례 공간 확보
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Segmentation Visualization - {file_name}", fontsize=16)

    # 서브플롯 1: 원본 이미지
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # 서브플롯 2: 컬러 마스크
    axes[1].imshow(color_mask_np)
    axes[1].set_title("Segmentation Mask", fontsize=14)
    axes[1].axis("off")

    # 서브플롯 3: 오버레이
    axes[2].imshow(overlay_np)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    # 범례를 오른쪽 중앙에 배치, 각 범례에 라벨 이름이 표시됨
    legend = fig.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),  # 오른쪽 여백 확보
        fontsize=12,
        title_fontsize=14
    )
    legend.get_frame().set_edgecolor("black")  # 범례 테두리 검은색 지정(선택 사항)

    # 레이아웃 조정: 서브플롯과 범례가 겹치지 않도록 여백 확보
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # plt.show()

    # (선택) 시각화 결과 저장
    out_fig_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '_visual.png')
    fig.savefig(out_fig_path)
    print(f"시각화 결과 저장: {out_fig_path}")