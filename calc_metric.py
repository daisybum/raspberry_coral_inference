import json
import os
import numpy as np
from PIL import Image, ImageDraw
from pycoral.utils import edgetpu
from pycoral.adapters import common, segment
from collections import defaultdict
from tqdm import tqdm

# 평가 메트릭 계산 함수 (TensorFlow 제거)
def compute_metrics(gt, pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    valid = (gt_flat < num_classes) & (pred_flat < num_classes)
    confusion_matrix += np.bincount(
        num_classes * gt_flat[valid].astype(np.int64) + pred_flat[valid].astype(np.int64),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    
    pixel_acc = np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-10)
    iou_per_class = np.diag(confusion_matrix) / (
        confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + 1e-10
    )
    dice_per_class = 2 * np.diag(confusion_matrix) / (
        confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) + 1e-10
    )
    miou = np.nanmean(iou_per_class)
    mean_dice = np.nanmean(dice_per_class)
    freq = confusion_matrix.sum(axis=1) / (confusion_matrix.sum() + 1e-10)
    fw_iou = (freq * iou_per_class).sum()
    
    return {
        'pixel_accuracy': pixel_acc,
        'mIoU': miou,
        'IoU_per_class': iou_per_class,
        'dice_per_class': dice_per_class,
        'fw_iou': fw_iou,
        'confusion_matrix': confusion_matrix
    }

# 경로 설정
annotations_path = '/workspace/merged_all/test_annotations.coco.json'
image_dir = '/workspace/merged_all'
model_path = 'model_quant_fixed_edgetpu.tflite'

# COCO JSON 로드
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

categories = {cat['id']: idx + 1 for idx, cat in enumerate(coco_data['categories'])}
category_names = ['background'] + [cat['name'] for cat in coco_data['categories']]
num_classes = len(categories) + 1

annotation_map = defaultdict(list)
for ann in coco_data['annotations']:
    annotation_map[ann['image_id']].append(ann)

# Edge TPU 모델 로드 (예외 처리 추가)
try:
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    print("Coral Edge TPU initialized successfully.")
except RuntimeError as e:
    print(f"Error initializing Coral Edge TPU: {e}")
    raise

input_width, input_height = common.input_size(interpreter)

# 전체 혼동 행렬 초기화
total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

# 이미지 처리
for image_info in tqdm(coco_data['images'], desc="Processing Images"):
    file_name = image_info['file_name']
    image_id = image_info['id']
    orig_width = image_info['width']
    orig_height = image_info['height']
    img_path = os.path.join(image_dir, file_name)
    
    if not os.path.exists(img_path):
        print(f"[WARN] 이미지 파일이 없음: {img_path}")
        continue
    
    img_pil = Image.open(img_path).convert('RGB')
    resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)
    
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    mask = segment.get_output(interpreter)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    
    mask_pil = Image.fromarray(mask.astype(np.uint8)).resize((orig_width, orig_height), resample=Image.NEAREST)
    pred_mask = np.array(mask_pil)
    
    gt_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    for ann in annotation_map[image_id]:
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                mask = Image.new('L', (orig_width, orig_height), 0)
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1, 2)).astype(int)
                    ImageDraw.Draw(mask).polygon([tuple(p) for p in poly], fill=1)
                mask = np.array(mask) * categories[ann['category_id']]
                gt_mask = np.maximum(gt_mask, mask)
            else:
                from pycocotools import mask as cocomask
                rle = ann['segmentation']
                mask = cocomask.decode(rle) * categories[ann['category_id']]
                gt_mask = np.maximum(gt_mask, mask)
    
    metrics = compute_metrics(gt_mask, pred_mask, num_classes)
    total_confusion_matrix += metrics['confusion_matrix']

# 클래스별 IoU와 Dice 계산
iou_per_class = np.diag(total_confusion_matrix) / (
    total_confusion_matrix.sum(axis=1) + total_confusion_matrix.sum(axis=0) - np.diag(total_confusion_matrix) + 1e-10
)
dice_per_class = 2 * np.diag(total_confusion_matrix) / (
    total_confusion_matrix.sum(axis=1) + total_confusion_matrix.sum(axis=0) + 1e-10
)
miou = np.nanmean(iou_per_class)
mean_dice = np.nanmean(dice_per_class)
pixel_acc = np.diag(total_confusion_matrix).sum() / (total_confusion_matrix.sum() + 1e-10)
freq = total_confusion_matrix.sum(axis=1) / (total_confusion_matrix.sum() + 1e-10)
fw_iou = (freq * iou_per_class).sum()

# 결과 출력
print("\n📊 최종 모델 성능 평가\n")
print("### 주요 성능 지표\n")
print("| 클래스      | IoU (Intersection over Union) | Dice 계수 |")
print("|-------------|-------------------------------|-----------|")
for cls, iou, dice in zip(category_names, iou_per_class, dice_per_class):
    print(f"| {cls:<11} | {iou:^25.4f} | {dice:^9.4f} |")
print(f"| **Mean**    | {miou:^25.4f} | {mean_dice:^9.4f} |")

print("\n### 추가 평가 지표")
print(f"- **Pixel Accuracy**: {pixel_acc:.4f}")
print(f"- **Frequency Weighted IoU**: {fw_iou:.4f}")

print("\n### 모델 개선 전후 비교\n")
print("| 성능 지표      | 초기 설정 | 개선 후 설정 |")
print("|----------------|-----------|--------------|")
print(f"| 전체 mIoU      | 약 0.42   | {miou:.4f}      |")
print(f"| Pixel Accuracy | 약 0.56   | {pixel_acc:.4f}      |")
