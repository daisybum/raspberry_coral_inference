#!/usr/bin/env python3
"""
experiments/sensor_label_dist.py
──────────────────────────────────────────────────────────────
• 여러 COCO json을 병합해 센서별(label_prefix) 클래스 분포 계산
• 결과:
    1) 터미널 표
    2) CSV  (sensor_label_dist.csv)
    3) PNG  (sensor_label_dist.png)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.logger import get_logger
from utils.file_utils import load_coco_annotations


log = get_logger("SensorEDA")


# ─────────────────────────────────────────────────────────────
# 센서명 추출 규칙
# ─────────────────────────────────────────────────────────────
def sensor_name(filename: str) -> str:
    """
    규칙:
    - MVW* 로 시작하면 앞 3 토큰 (MVW_B1_000003)
    - 그 외는 앞 4 토큰
    """
    stem = Path(filename).stem
    tokens = stem.split("_")
    return "_".join(tokens[:3] if stem.startswith("MVW") else tokens[:4])


# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="센서별 라벨 분포 EDA (COCO 형식)"
    )
    p.add_argument(
        "jsons",
        nargs="+",
        help="분석할 COCO json 경로 (여러 개)",
    )
    p.add_argument(
        "--out-dir",
        default="experiments",
        help="CSV/PNG 저장 폴더 (default: experiments)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON 병합
    combined = {"images": [], "annotations": [], "categories": []}
    for j in args.jsons:
        coco = load_coco_annotations(j)
        combined["images"].extend(coco["images"])
        combined["annotations"].extend(coco["annotations"])
        if not combined["categories"]:
            combined["categories"] = coco["categories"]
        log.info(f"✅  loaded {j}  (imgs {len(coco['images'])})")

    cat_id2name = {c["id"]: c["name"] for c in combined["categories"]}

    # ── 이미지 id → 센서명
    img2sensor = {
        img["id"]: sensor_name(img["file_name"]) for img in combined["images"]
    }

    # ── 센서별 카운트
    sensor_cnt: dict[str, Counter[int]] = defaultdict(Counter)
    for ann in combined["annotations"]:
        sensor = img2sensor.get(ann["image_id"])
        if sensor:
            sensor_cnt[sensor][ann["category_id"]] += 1

    # ── DataFrame (행=sensor, 열=라벨명, 값=퍼센트)
    df = (
        pd.DataFrame(sensor_cnt)
        .T.fillna(0)
        .astype(int)
        .rename(columns=cat_id2name)
    )
    df_pct = df.div(df.sum(axis=1), axis=0) * 100
    df_pct.sort_index(inplace=True)

    # ── 출력
    print("\n📊  센서별 라벨 분포 (%)\n")
    print(df_pct.round(2).to_string())
    csv_path = out_dir / "sensor_label_dist.csv"
    df_pct.to_csv(csv_path, float_format="%.2f")
    log.info(f"💾  CSV 저장 → {csv_path}")

    # ── 바 차트 저장
    plt.figure(figsize=(max(8, len(df_pct)), 6))
    df_pct.plot.bar(stacked=True, figsize=(max(8, len(df_pct)), 6))
    plt.ylabel("Percentage (%)")
    plt.title("Sensor‑wise Label Distribution")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    png_path = out_dir / "sensor_label_dist.png"
    plt.savefig(png_path)
    plt.close()
    log.info(f"🖼  PNG 저장 → {png_path}")


if __name__ == "__main__":
    main()
