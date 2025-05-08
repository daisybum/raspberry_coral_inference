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
    preprocess_for_model,
    resize_mask,
    colorize_mask,
    blend_mask,
)
from utils.visualization import visualize_and_save


# ─────────────────────────────────────────────────────────────
def run_camera(cfg: Dict[str, Any], logger, interval: int = 30):
    """
    libcamera-still 로  `interval` 초마다 이미지를 캡처한 뒤
    추론·시각화 PNG를 `paths.output_dir` 에 저장한다.
    """
    pipe = SegmentationPipeline(cfg, skip_visualize=True)
    interp = pipe.interpreter
    in_w, in_h = pipe.in_w, pipe.in_h
    palette = pipe.palette
    legend = pipe.legend_patches

    cap_dir = "./captured_images"
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(cap_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"[CAMERA] Camera mode started - capture interval: {interval}s")

    while True:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"cap_{ts}.jpg"
        cap_path = os.path.join(cap_dir, fn)

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

        # ── 추론 & 시각화
        img = load_image(cap_path)
        resized = preprocess_for_model(img, (in_w, in_h))

        from pycoral.adapters import common, segment

        common.set_input(interp, resized)
        interp.invoke()
        raw_mask = segment.get_output(interp)
        if raw_mask.ndim == 3:
            raw_mask = np.argmax(raw_mask, axis=-1)

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
        logger.info(f"[CAMERA] Image processed and saved to: {out_path}")
        time.sleep(interval)
