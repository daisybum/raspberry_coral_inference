"""
camera.py
──────────────────────────────────────────────────────────────
실시간 라즈베리파이 카메라 캡처 → Edge‑TPU 추론 → PNG 저장
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, Any

import numpy as np
from PIL import Image

from pipeline import SegmentationPipeline
from utils.image_utils import (
    load_image,
    resize_mask,
    colorize_mask,
    blend_mask,
)
from utils.visualization import visualize_and_save


# ─────────────────────────────────────────────────────────────
def run_camera(cfg: Dict[str, Any], logger, interval: int = 30, save_captured: bool = False, delete_after: bool = False):
    """
    libcamera-still 로 `interval` 초마다 이미지를 캡처한 뒤
    추론·시각화 PNG를 `paths.output_dir` 에 저장한다.
    
    Parameters:
        cfg: 설정 딕셔너리
        logger: 로거 인스턴스
        interval: 캡처 간격(초)
        save_captured: 캡처된 이미지를 저장할지 여부
        delete_after: 추론 후 캡처된 이미지를 삭제할지 여부
    """
    pipe = SegmentationPipeline(cfg, skip_visualize=True)
    palette = pipe.palette
    legend = pipe.legend_patches

    cap_dir = "./captured_images"
    out_dir = cfg["paths"]["output_dir"]
    
    # 캡처 이미지를 저장하는 경우에만 디렉토리 생성
    if save_captured:
        os.makedirs(cap_dir, exist_ok=True)
    
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[CAMERA] Camera mode started - capture interval: {interval}s")
    logger.info(f"[CAMERA] Save captured images: {save_captured}, Delete after inference: {delete_after}")

    while True:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"cap_{ts}.jpg"
        cap_path = os.path.join(cap_dir, fn) if save_captured else "/tmp/temp_capture.jpg"

        # libcamera-still 무음 캡처
        cmd = [
            "libcamera-still",
            "-n",
            "-o",
            cap_path,
            "--width",
            "1640",
            "--height",
            "1232",
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.isfile(cap_path):
            logger.warning("[CAMERA] Image capture failed"); time.sleep(interval); continue

        # ── 추론 & 시각화 (SegmentationPipeline 이용 – CPU)
        img = load_image(cap_path)

        raw_mask = pipe._infer_mask(img)  # sensor 정보 없음

        mask_full = resize_mask(raw_mask, img.size)  # img.size = (W,H)
        color_mask = colorize_mask(mask_full, palette)
        overlay = blend_mask(np.array(img), mask_full, palette)

        out_path = visualize_and_save(
            fn,
            np.array(img),
            color_mask,
            overlay,
            legend,
            out_dir,
            save_image=False
        )
        
        # 추론 후 이미지 삭제 (save_captured가 True이고 delete_after가 True인 경우)
        if save_captured and delete_after and os.path.isfile(cap_path):
            try:
                os.remove(cap_path)
                logger.info(f"[CAMERA] Deleted captured image: {cap_path}")
            except Exception as e:
                logger.warning(f"[CAMERA] Failed to delete captured image: {e}")
        
        # 임시 파일 삭제 (save_captured가 False인 경우)
        elif not save_captured and os.path.isfile(cap_path):
            try:
                os.remove(cap_path)
            except Exception:
                pass
                
        logger.info(f"[CAMERA] Image processed and saved to: {out_path}")
        time.sleep(interval)
