"""
pipeline.py
──────────────────────────────────────────────────────────────
Edge‑TPU SegmentationPipeline

• 전처리 → 추론 → 후처리 → (선택)시각화까지 한 번에 수행
• utils/ 패키지에 의존
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, List

import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

from utils.image_utils import (
    load_image,
    preprocess_for_model,
    resize_mask,
    colorize_mask,
    blend_mask,
)
from utils.visualization import (
    create_legend_patches,
    visualize_and_save,
)
from utils.file_utils import image_infos_generator
from utils.logger import get_logger
from utils.timer import elapsed


class SegmentationPipeline:
    """
    cfg : config.yaml 로드 결과(dict)
    skip_visualize : True → PNG 저장 생략
    """

    def __init__(self, cfg: Dict[str, Any], *, skip_visualize: bool = False):
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.palette = np.array(cfg["palette"], dtype=np.uint8)
        self.class_names: List[str] = cfg["class_names"]
        self.skip_visualize = skip_visualize

        # 로거
        self.logger = get_logger("SegPipeline")

        # Edge‑TPU Interpreter
        self.interpreter = edgetpu.make_interpreter(self.paths["model"])
        self.interpreter.allocate_tensors()
        self.in_w, self.in_h = common.input_size(self.interpreter)
        self.logger.info(f"[PIPELINE] Interpreter initialized successfully - input dimensions: {self.in_w}x{self.in_h}")

        # 범례 패치
        self.legend_patches = create_legend_patches(
            self.palette, self.class_names
        )

        # 출력 폴더
        os.makedirs(self.paths["output_dir"], exist_ok=True)

    # ──────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────
    def _infer_mask(self, pil_img):
        """Edge‑TPU 추론 → 2D mask(uint8) 반환"""
        common.set_input(self.interpreter, pil_img)
        self.interpreter.invoke()
        output = segment.get_output(self.interpreter)
        if output.ndim == 3:
            output = np.argmax(output, axis=-1)
        return output.astype(np.uint8)

    def _process_one(self, info: Dict[str, Any]) -> float:
        """
        단일 이미지 처리. (PNG 저장은 skip_visualize에 따라)
        반환값 : 처리 시간(초)
        """
        t0 = time.time()
        img_path = os.path.join(self.paths["image_dir"], info["file_name"])
        pil_img = load_image(img_path)

        # 전처리 & 추론
        resized = preprocess_for_model(pil_img, (self.in_w, self.in_h))
        raw_mask = self._infer_mask(resized)

        # 후처리
        mask_full = resize_mask(raw_mask, (info["width"], info["height"]))
        color_mask = colorize_mask(mask_full, self.palette)
        orig_np = np.array(
            pil_img.resize((info["width"], info["height"]))
        )
        overlay = blend_mask(orig_np, mask_full, self.palette)

        # 시각화 저장
        if not self.skip_visualize:
            visualize_and_save(
                info["file_name"],
                orig_np,
                color_mask,
                overlay,
                self.legend_patches,
                self.paths["output_dir"],
                save_image=False
            )

        return time.time() - t0

    # ──────────────────────────────────────────────
    # Public run
    # ──────────────────────────────────────────────
    def run(self, coco_dict: Dict[str, Any]):
        total = len(coco_dict["images"])
        processed, elapsed_sum = 0, 0.0

        with elapsed("Dataset Inference", "SegPipeline"):
            for info in image_infos_generator(
                coco_dict, self.paths["image_dir"]
            ):
                dt = self._process_one(info)
                processed += 1
                elapsed_sum += dt
                self.logger.info(
                    f"[PIPELINE] Processing image {processed}/{total}: {info['file_name']} - inference time: {dt:.2f}s"
                )

        if processed:
            avg = elapsed_sum / processed
            self.logger.info(
                f"[PIPELINE] Processing complete: {processed}/{total} images processed, average time per image: {avg:.2f}s"
            )
