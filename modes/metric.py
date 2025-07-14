"""
metric.py
──────────────────────────────────────────────────────────────
COCO GT 와 예측 마스크를 비교해
PixelAcc · mIoU · Dice · FW IoU 등을 계산.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

from pipeline import SegmentationPipeline
from utils.file_utils import load_coco_annotations
from utils.image_utils import (
    load_image,
    preprocess_for_model,
    resize_mask,
)
from pipeline import _sensor_to_vec

# ─────────────────────────────────────────────────────────────
def _confusion(gt: np.ndarray, pred: np.ndarray, n_cls: int):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    gt_f, pr_f = gt.flatten(), pred.flatten()
    valid = (gt_f < n_cls) & (pr_f < n_cls)
    idx = n_cls * gt_f[valid] + pr_f[valid]
    cm += np.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    return cm


def _metrics(cm: np.ndarray):
    diag = np.diag(cm)
    pixel_acc = diag.sum() / (cm.sum() + 1e-10)
    iou = diag / (cm.sum(1) + cm.sum(0) - diag + 1e-10)
    dice = 2 * diag / (cm.sum(1) + cm.sum(0) + 1e-10)
    miou, mdice = np.nanmean(iou), np.nanmean(dice)
    freq = cm.sum(1) / (cm.sum() + 1e-10)
    fwiou = (freq * iou).sum()
    return pixel_acc, miou, mdice, iou, dice, fwiou


# ─────────────────────────────────────────────────────────────
def run_metric(cfg: Dict[str, Any], logger):
    """
    confusion matrix → PixelAcc, mIoU, Dice, FW IoU 출력
    """
    pipe = SegmentationPipeline(cfg, skip_visualize=True)
    # pipe 내부 interpreter는 직접 사용하지 않고 helper 호출

    coco = load_coco_annotations(cfg["paths"]["annotations"])

    # category id → 연속 인덱스(1부터)
    cat_map = {c["id"]: idx + 1 for idx, c in enumerate(coco["categories"])}
    class_names = ["background"] + [c["name"] for c in coco["categories"]]
    n_cls = len(class_names)

    total_cm = np.zeros((n_cls, n_cls), dtype=np.int64)

    for info in tqdm(coco["images"], desc="Metric"):
        img_path = os.path.join(cfg["paths"]["image_dir"], info["file_name"])
        if not os.path.isfile(img_path):
            continue

        # ── 예측 마스크
        img = load_image(img_path)
        sensor_vec = _sensor_to_vec(info.get("sensor_info", {}))
        raw_mask = pipe._infer_mask(img, sensor_vec)
        pred_mask = resize_mask(raw_mask, (info["width"], info["height"]))

        # ── GT 마스크
        gt_mask = np.zeros_like(pred_mask)
        for ann in (a for a in coco["annotations"] if a["image_id"] == info["id"]):
            if "segmentation" not in ann:
                continue
            if isinstance(ann["segmentation"], list):
                tmp = Image.new("L", (info["width"], info["height"]), 0)
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((-1, 2)).astype(int)
                    ImageDraw.Draw(tmp).polygon([tuple(p) for p in poly], fill=1)
                gt_mask = np.maximum(gt_mask, np.array(tmp) * cat_map[ann["category_id"]])
            else:
                from pycocotools import mask as cocomask
                gt_mask = np.maximum(
                    gt_mask,
                    cocomask.decode(ann["segmentation"]) * cat_map[ann["category_id"]],
                )

        total_cm += _confusion(gt_mask, pred_mask, n_cls)

    pa, miou, mdice, iou, dice, fwiou = _metrics(total_cm)

    print("[METRIC] Final evaluation results:")
    print("| Class       | IoU  | Dice |")
    print("|-------------|------|------|")
    for n, i, d in zip(class_names, iou, dice):
        print(f"| {n:<11} | {i:.4f} | {d:.4f} |")
    print(f"\nMean IoU  : {miou:.4f}")
    print(f"Mean Dice : {mdice:.4f}")
    print(f"PixelAcc  : {pa:.4f}")
    print(f"FW IoU    : {fwiou:.4f}")

    logger.info("[METRIC] Metric calculation completed successfully")
