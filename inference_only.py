import json
import os
import numpy as np
from PIL import Image
from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# 0) 경로 설정
annotations_path = '/workspace/merged_all/test_annotations.coco.json'
image_dir = '/workspace/merged_all'
model_path = 'model_quant_fixed_edgetpu.tflite'

# COCO 형식 JSON 파일 로드
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# 1) Edge TPU용 TFLite 모델 로드 및 인터프리터 초기화
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# 2) 모델 입력 크기 확인
input_width, input_height = common.input_size(interpreter)
print(f"모델 입력 크기: {input_width} x {input_height}")

# 3) COCO 데이터에 있는 이미지 순회하며 추론 실행
for image_info in coco_data['images']:
    file_name = image_info['file_name']
    orig_width = image_info['width']
    orig_height = image_info['height']
    img_path = os.path.join(image_dir, file_name)
    
    if not os.path.exists(img_path):
        print(f"[WARN] 이미지 파일이 없음: {img_path}")
        continue

    print(f"처리 중: {img_path}")
    
    # (1) 이미지 로드 및 전처리 (모델 입력 크기로 리사이즈)
    img_pil = Image.open(img_path).convert('RGB')
    resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)
    
    # (2) 추론 수행
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    mask = segment.get_output(interpreter)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    
    # (3) 원본 이미지 크기로 결과 마스크 리사이즈 (선택 사항)
    # mask_pil = Image.fromarray(mask.astype(np.uint8)).resize((orig_width, orig_height), resample=Image.NEAREST)
    # mask_np = np.array(mask_pil)
    
    print(f"{file_name} 추론 완료")
