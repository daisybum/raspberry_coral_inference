# inference.py

import os
import json
import time
import logging
import argparse
import numpy as np
import yaml

# GUI 없는 환경용 백엔드
import matplotlib
matplotlib.use('Agg')

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment
from PIL import Image

from utils.image_utils import (
    load_image, resize_image, resize_mask,
    colorize_mask, blend_mask
)
from utils.visualization import create_legend_patches, visualize_and_save


def setup_logger():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_interpreter(model_path: str):
    interp = edgetpu.make_interpreter(model_path)
    interp.allocate_tensors()
    return interp


def main():
    parser = argparse.ArgumentParser(description="Edge TPU Segmentation Inference")
    parser.add_argument(
        "--config", default="config.yaml",
        help="YAML 형식의 설정 파일 경로"
    )
    args = parser.parse_args()

    setup_logger()
    cfg = load_config(args.config)

    # 설정 불러오기
    paths       = cfg["paths"]
    palette     = np.array(cfg["palette"], dtype=np.uint8)
    class_names = cfg["class_names"]

    # 인터프리터 초기화
    logging.info("모델 로드 및 Interpreter 초기화")
    interpreter = build_interpreter(paths["model"])
    in_w, in_h = common.input_size(interpreter)
    logging.info(f"Model input size: {in_w} x {in_h}")

    # COCO 주석 로드
    with open(paths["annotations"], "r") as f:
        coco = json.load(f)

    # 범례 패치 생성
    legend = create_legend_patches(palette, class_names)

    # 출력 디렉토리
    os.makedirs(paths["output_dir"], exist_ok=True)

    total_start = time.time()
    processed = 0
    total_images = len(coco["images"])

    for info in coco["images"]:
        fn = info["file_name"]
        img_path = os.path.join(paths["image_dir"], fn)
        if not os.path.isfile(img_path):
            logging.warning(f"파일 없음: {img_path}")
            continue

        t0 = time.time()
        # 원본 이미지 로드 & 리사이즈
        img = load_image(img_path)
        resized = resize_image(img, (in_w, in_h))

        # 추론
        common.set_input(interpreter, resized)
        interpreter.invoke()
        raw_mask = segment.get_output(interpreter)
        if raw_mask.ndim == 3:
            raw_mask = np.argmax(raw_mask, axis=-1)
        raw_mask = raw_mask.astype(np.uint8)

        # 마스크 원본 크기로 복원
        mask_full = resize_mask(raw_mask, (info["width"], info["height"]))
        # 컬러 마스크 & 오버레이 생성
        color_mask = colorize_mask(mask_full, palette)
        orig_np = np.array(resize_image(img, (info["width"], info["height"])))
        overlay = blend_mask(orig_np, mask_full)

        # 시각화 및 저장
        out_path = visualize_and_save(
            fn, orig_np, color_mask, overlay,
            legend, paths["output_dir"]
        )

        dt = time.time() - t0
        processed += 1
        logging.info(f"[{processed}/{total_images}] {fn} → {dt:.2f}s, 저장: {out_path}")

    total_dt = time.time() - total_start
    avg_dt = total_dt / processed if processed else 0
    logging.info(f"완료: {processed}장, 총 {total_dt:.2f}s, 평균 {avg_dt:.2f}s")


if __name__ == "__main__":
    main()
