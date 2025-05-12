"""
stress_test.py
──────────────────────────────────────────────────────────────
data 폴더 내 이미지를 무한반복으로 추론하는 스트레스 테스트

• 주요 기능:
  - data 폴더 내 이미지 파일 자동 탐색
  - 무한 반복 추론 (Ctrl+C로 중단)
  - 추론 시간 통계 수집 및 출력
"""

from __future__ import annotations

import os
import time
import glob
from typing import Dict, Any, List
import statistics

import numpy as np

from pipeline import SegmentationPipeline
from utils.image_utils import load_image, preprocess_for_model
from utils.timer import elapsed


def get_image_files(data_dir: str) -> List[str]:
    """데이터 디렉토리에서 이미지 파일 목록 가져오기"""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    return sorted(image_files)


def process_image(pipeline: SegmentationPipeline, image_path: str) -> float:
    """
    단일 이미지 추론 처리
    반환값: 처리 시간(초)
    """
    t0 = time.time()
    
    # 이미지 로드 및 정보 생성
    pil_img = load_image(image_path)
    file_name = os.path.basename(image_path)
    width, height = pil_img.size
    
    # 전처리 & 추론
    in_w, in_h = pipeline.in_w, pipeline.in_h
    resized = preprocess_for_model(pil_img, (in_w, in_h))
    raw_mask = pipeline._infer_mask(resized)
    
    # 후처리 (시각화 없이 마스크만 생성)
    from utils.image_utils import resize_mask
    mask_full = resize_mask(raw_mask, (width, height))
    
    return time.time() - t0


def run_stress_test(cfg: Dict[str, Any], logger, interval: int = 0, iterations: int = None):
    """
    데이터 폴더 내 이미지를 무한 반복 추론하는 스트레스 테스트
    
    Parameters:
        cfg: 설정 정보
        logger: 로거 인스턴스
        interval: 이미지 처리 간 대기 시간(초), 0이면 대기 없음
        iterations: 반복 횟수 제한, None이면 무한 반복
    """
    # 데이터 폴더 경로
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    logger.info(f"Starting stress test on images in {data_dir}")
    
    # 파이프라인 초기화 (시각화 생략)
    pipeline = SegmentationPipeline(cfg, skip_visualize=True)
    
    # 이미지 파일 목록 가져오기
    image_files = get_image_files(data_dir)
    if not image_files:
        logger.error(f"No image files found in {data_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images in {data_dir}")
    
    # 통계 수집용 변수
    iteration_count = 0
    total_inferences = 0
    inference_times = []
    start_time = time.time()
    
    try:
        while iterations is None or iteration_count < iterations:
            iteration_count += 1
            logger.info(f"Starting iteration {iteration_count}")
            
            iteration_times = []
            for idx, image_path in enumerate(image_files):
                file_name = os.path.basename(image_path)
                logger.info(f"Processing image {idx+1}/{len(image_files)}: {file_name}")
                
                try:
                    dt = process_image(pipeline, image_path)
                    inference_times.append(dt)
                    iteration_times.append(dt)
                    total_inferences += 1
                    
                    logger.info(f"Inference time: {dt*1000:.2f} ms")
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")
                
                # 다음 이미지 처리 전 대기
                if interval > 0:
                    time.sleep(interval)
            
            # 반복 완료 후 통계 출력
            if iteration_times:
                avg_time = statistics.mean(iteration_times) * 1000
                max_time = max(iteration_times) * 1000
                min_time = min(iteration_times) * 1000
                logger.info(
                    f"Iteration {iteration_count} stats: "
                    f"Avg={avg_time:.2f}ms, Min={min_time:.2f}ms, Max={max_time:.2f}ms"
                )
    
    except KeyboardInterrupt:
        logger.info("Stress test stopped by user")
    
    # 최종 통계 출력
    elapsed_time = time.time() - start_time
    if inference_times:
        avg_time = statistics.mean(inference_times) * 1000
        max_time = max(inference_times) * 1000
        min_time = min(inference_times) * 1000
        std_dev = statistics.stdev(inference_times) * 1000 if len(inference_times) > 1 else 0
        
        logger.info("=" * 50)
        logger.info(f"Stress Test Summary:")
        logger.info(f"Total runtime: {elapsed_time:.2f} seconds")
        logger.info(f"Total iterations: {iteration_count}")
        logger.info(f"Total inferences: {total_inferences}")
        logger.info(f"Average inference time: {avg_time:.2f} ms")
        logger.info(f"Min inference time: {min_time:.2f} ms")
        logger.info(f"Max inference time: {max_time:.2f} ms")
        logger.info(f"Standard deviation: {std_dev:.2f} ms")
        logger.info(f"Throughput: {total_inferences/elapsed_time:.2f} inferences/second")
        logger.info("=" * 50)
