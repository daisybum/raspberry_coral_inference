#!/usr/bin/env python3
"""
experiments/stress_test.py
──────────────────────────────────────────────────────────────
• 30 초마다 libcamera-still 캡처 → Edge‑TPU 추론 → PNG 저장
• CPU 온도·추론 시간·오류를 지속적으로 로깅
• utils/  패키지 + pipeline.SegmentationPipeline  활용
"""

from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
from PIL import Image

from utils.logger import get_logger
from utils.image_utils import (
    preprocess_for_model,
    resize_mask,
    colorize_mask,
    blend_mask,
)
from utils.visualization import visualize_and_save, create_legend_patches
from pipeline import SegmentationPipeline


# ─────────────────────────────────────────────────────────────
# 설정 읽기 + 기본 경로
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]         # 프로젝트 루트
CFG_PATH = ROOT / "config.yaml"

with open(CFG_PATH, "r") as f:
    CFG: Dict[str, Any] = yaml.safe_load(f)

CAPTURE_DIR = Path("/media/pi/ESD-USB/captured_images")
OUTPUT_DIR  = Path("/media/pi/ESD-USB/output_visual")
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True,  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 로거
# ─────────────────────────────────────────────────────────────
log = get_logger("StressTest")

# ─────────────────────────────────────────────────────────────
# 파이프라인 & 팔레트 준비
# ─────────────────────────────────────────────────────────────
pipe = SegmentationPipeline(CFG, skip_visualize=True)   # 모델·interpreter 초기화
interp   = pipe.interpreter
in_w, in_h = pipe.in_w, pipe.in_h
palette  = pipe.palette
legend   = pipe.legend_patches


# ─────────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────────
def cpu_temperature() -> float | None:
    """라즈베리파이 CPU 온도(℃) 반환. 실패 시 None."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"], text=True
        )  # 예:  temp=45.2'C
        return float(out.split("=")[1].replace("'C", ""))
    except Exception as e:
        log.warning(f"temp read fail: {e}")
        return None


def capture_image() -> Path | None:
    """libcamera‑still 로 이미지를 캡처해 경로를 리턴."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    img_path = CAPTURE_DIR / f"cap_{ts}.jpg"
    cmd = [
        "libcamera-still",
        "-n",
        "-o",
        str(img_path),
        "--width",
        "1640",
        "--height",
        "1232",
    ]
    rtn = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rtn.returncode == 0 and img_path.exists():
        return img_path
    log.warning("capture failed")
    return None


# ─────────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────────
INTERVAL = 30  # seconds

log.info("🚀  Coral Stress‑Test 시작")

while True:
    # ── 1) 시스템 상태 로깅
    temp = cpu_temperature()
    if temp is not None:
        log.info(f"🌡  CPU Temp: {temp:.1f}°C")

    # ── 2) 이미지 캡처
    img_path = capture_image()
    if img_path is None:
        time.sleep(INTERVAL)
        continue

    try:
        # ── 3) 추론
        t0 = time.time()
        pil = Image.open(img_path).convert("RGB")
        resized = preprocess_for_model(pil, (in_w, in_h))

        from pycoral.utils import common, segment

        common.set_input(interp, resized)
        interp.invoke()
        raw_mask = segment.get_output(interp)
        if raw_mask.ndim == 3:
            raw_mask = np.argmax(raw_mask, axis=-1)

        # ── 4) 후처리 & 시각화
        mask_full = resize_mask(raw_mask, pil.size)  # (W,H)
        color_mask = colorize_mask(mask_full, palette)
        overlay = blend_mask(np.array(pil), mask_full, palette)

        out_path = visualize_and_save(
            img_path.stem,
            np.array(pil),
            color_mask,
            overlay,
            legend,
            str(OUTPUT_DIR),
        )

        dt = time.time() - t0
        log.info(f"✅  {img_path.name}  infer+save  {dt:.2f}s → {out_path}")

    except Exception as e:
        log.error(f"❌  오류: {e}")

    # ── 5) 대기
    log.info(f"🕒  {INTERVAL}s 대기 후 다음 캡처")
    time.sleep(INTERVAL)
