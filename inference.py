#!/usr/bin/env python3
"""
inference.py
──────────────────────────────────────────────────────────────
공통 CLI 엔트리포인트

--mode {bench|visual|camera|metric} 로 실행 기능 선택
"""

from __future__ import annotations

import argparse
import yaml

from utils.logger import get_logger
from modes.bench_and_visual import run_bench, run_visual
from modes.camera import run_camera
from modes.metric import run_metric

# 모드 → 실행 함수 매핑
MODE_TABLE = {
    "bench": run_bench,
    "visual": run_visual,
    "camera": run_camera,
    "metric": run_metric,
}


# ─────────────────────────────────────────────────────────────
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Edge‑TPU Segmentation – 통합 실행기"
    )
    parser.add_argument("--config", default="config.yaml", help="YAML 설정 파일")
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODE_TABLE.keys(),
        help="실행 모드 선택",
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="camera 모드 캡처 간격(초)"
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    logger = get_logger("Main")

    # 선택한 모드 실행
    if args.mode in ("bench", "visual", "metric"):
        MODE_TABLE[args.mode](cfg, logger)
    elif args.mode == "camera":
        MODE_TABLE["camera"](cfg, logger, args.interval)


if __name__ == "__main__":
    main()
