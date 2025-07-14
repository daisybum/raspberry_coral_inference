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

# ──────────────────────────────────────────────────────────────
# Raspberry Pi 4 CPU (TFLite) 환경 지원
# pycoral → tflite-runtime 로 대체, Edge-TPU delegate 제거
# ──────────────────────────────────────────────────────────────
# tflite-runtime 모듈이 우선, 미설치 시 tensorflow.lite 로 폴백
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ModuleNotFoundError:
    import tensorflow as tf  # type: ignore

    tflite = tf.lite  # pyright: ignore

# 멀티스레드 옵션 (환경변수로 제어)
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))

from PIL import Image
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

        # ── TFLite Interpreter (CPU)
        self.interpreter = tflite.Interpreter(
            model_path=self.paths["model"], num_threads=NUM_THREADS
        )
        self.interpreter.allocate_tensors()

        # 입력/출력 텐서 메타데이터
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 이미지 & 센서 입력 인덱스 자동 식별
        self.img_input = next(
            (d for d in self.input_details if len(d["shape"]) >= 3),
            self.input_details[0],
        )
        self.sensor_input = next(
            (d for d in self.input_details if d is not self.img_input), None
        )

        self.in_w, self.in_h = self.img_input["shape"][2], self.img_input["shape"][1]

        self.logger.info(
            f"[PIPELINE] TFLite interpreter (CPU) initialized – input {self.in_w}x{self.in_h}, "
            f"sensor_input: {self.sensor_input is not None}"
        )

        # 범례 패치
        self.legend_patches = create_legend_patches(
            self.palette, self.class_names
        )

        # 출력 폴더
        os.makedirs(self.paths["output_dir"], exist_ok=True)

    # ──────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────
    def _infer_mask(self, pil_img, sensor_vec: np.ndarray | None = None):  # type: ignore
        """TFLite CPU 추론 → 2D mask(uint8) 반환"""

        # 이미지 입력 설정
        img_arr = _prepare_input(pil_img, self.img_input)
        self.interpreter.set_tensor(self.img_input["index"], img_arr)

        # (선택) 센서 입력 설정
        if self.sensor_input is not None:
            if sensor_vec is None:
                sensor_vec = np.zeros(self.sensor_input["shape"], dtype=np.float32)
            else:
                if sensor_vec.shape != tuple(self.sensor_input["shape"]):
                    sensor_vec = sensor_vec.reshape(self.sensor_input["shape"])

            self.interpreter.set_tensor(
                self.sensor_input["index"], sensor_vec.astype(self.sensor_input["dtype"])  # type: ignore[attr-defined]
            )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        if output.ndim == 4:  # (1, H, W, C)
            output = output[0]
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
        sensor_vec = _sensor_to_vec(info.get("sensor_info", {}))
        raw_mask = self._infer_mask(resized, sensor_vec)

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

# ──────────────────────────────────────────────────────────────
# Sensor helpers
# ──────────────────────────────────────────────────────────────


def _sensor_to_vec(sensor_data: dict | list | None) -> np.ndarray:  # type: ignore
    """annotation JSON의 sensor_info → (1, 6) float32 벡터 변환 & 0 – 255 스케일링"""

    if not sensor_data:
        return np.zeros((1, 6), dtype=np.float32)
    if isinstance(sensor_data, list):
        sensor_data = sensor_data[0] if sensor_data else {}

    keys = [
        "objectTemp",
        "humi",
        "pressure",
        "latitude",
        "longitude",
        "height",
    ]
    vec = np.asarray([float(sensor_data.get(k, 0.0)) for k in keys], dtype=np.float32)

    # 학습·양자화와 동일한 0~255 스케일 정규화
    mins = np.asarray([-100, 0, 950, -90, -180, 0], dtype=np.float32)
    maxs = np.asarray([100, 100, 1050, 90, 180, 1000], dtype=np.float32)
    vec = np.clip(vec, mins, maxs)
    vec = (vec - mins) * 255.0 / (maxs - mins)

    return np.asarray([vec], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Image input helper (dtype/scale 대응)
# ──────────────────────────────────────────────────────────────


def _prepare_input(pil: Image.Image, input_info: Dict[str, Any]) -> np.ndarray:  # type: ignore
    """PIL 이미지를 Interpreter 기대 형식으로 변환 (1, H, W, C)."""

    w, h = input_info["shape"][2], input_info["shape"][1]
    img = pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img)

    dtype = input_info["dtype"]
    if dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

    return np.expand_dims(arr, axis=0)
