#!/usr/bin/env python3
"""
inference.py
──────────────────────────────────────────────────────────────
공통 CLI 엔트리포인트 (라즈베리파이 4 CPU 실행)

--mode {bench|visual|camera|metric|stress} 로 실행 기능 선택
"""

from __future__ import annotations

import argparse
import yaml
import os

from utils.logger import get_logger
from modes.bench_and_visual import run_bench, run_visual
from modes.camera import run_camera
from modes.metric import run_metric
from modes.stress_test import run_stress_test

# 모드 → 실행 함수 매핑
MODE_TABLE = {
    "bench": run_bench,
    "visual": run_visual,
    "camera": run_camera,
    "metric": run_metric,
    "stress": run_stress_test,
}


# ─────────────────────────────────────────────────────────────
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Edge-TPU Segmentation - Unified Runner"
    )
    parser.add_argument("--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODE_TABLE.keys(),
        help="Select execution mode",
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Capture interval (seconds) for camera/stress mode"
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Number of iterations for stress test (None=infinite)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization for stress test mode"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of CPU threads to use (overrides NUM_THREADS env)",
    )
    args = parser.parse_args()

    # ── 스레드 수 환경변수 적용 (SegmentationPipeline 생성 전에!)
    if args.num_threads is not None and args.num_threads > 0:
        import pipeline as pl

        os.environ["NUM_THREADS"] = str(args.num_threads)
        pl.NUM_THREADS = args.num_threads  # type: ignore[attr-defined]

    cfg = load_cfg(args.config)
    logger = get_logger("Main")
    logger.info(f"[MAIN] Starting application in {args.mode} mode with config from {args.config}")

    # 선택한 모드 실행
    if args.mode in ("bench", "visual", "metric"):
        MODE_TABLE[args.mode](cfg, logger)
    elif args.mode == "camera":
        MODE_TABLE["camera"](cfg, logger, args.interval)
    elif args.mode == "stress":
        MODE_TABLE["stress"](cfg, logger, args.interval, args.iterations, args.visualize)


if __name__ == "__main__":
    main()
