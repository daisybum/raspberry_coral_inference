"""
bench_and_visual.py
──────────────────────────────────────────────────────────────
• run_bench  : COCO 이미지셋 추론 속도 측정 (PNG 저장 없음)
• run_visual : COCO 이미지셋 추론 & 시각화 PNG 저장
"""

from __future__ import annotations

import time
from typing import Dict, Any

from pipeline import SegmentationPipeline
from utils.file_utils import load_coco_annotations, image_infos_generator


# ─────────────────────────────────────────────────────────────
def run_bench(cfg: Dict[str, Any], logger):
    """
    이미지별 추론 시간을 로그로 남기고 평균 속도를 출력.
    """
    pipe = SegmentationPipeline(cfg, skip_visualize=True)
    coco = load_coco_annotations(cfg["paths"]["annotations"])

    total, elapsed_sum = 0, 0.0
    for info in image_infos_generator(coco, cfg["paths"]["image_dir"]):
        t0 = time.time()
        pipe._process_one(info)          # 시각화 생략
        dt = time.time() - t0
        elapsed_sum += dt
        total += 1
        logger.info(f"[BENCH] Processing image: {info['file_name']:<40} inference time: {dt*1000:7.2f} ms")

    if total:
        logger.info(
            f"[BENCH] Benchmark complete - Average inference time: {elapsed_sum/total*1000:.2f} ms "
            f"across {total} images"
        )


# ─────────────────────────────────────────────────────────────
def run_visual(cfg: Dict[str, Any], logger):
    """
    전체 이미지 추론 후 `paths.output_dir` 아래에
    `_visual.png` 파일을 저장한다.
    """
    pipe = SegmentationPipeline(cfg, skip_visualize=False)
    coco = load_coco_annotations(cfg["paths"]["annotations"])
    pipe.run(coco)
    logger.info("[VISUAL] Visualization process completed successfully")
