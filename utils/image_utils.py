"""
이미지 I/O‧전처리 및 마스크 후처리 모음
Author : 상현님 프로젝트
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image


# ────────────────────────────────────────────────────────────
# 기본 이미지 입출력
# ────────────────────────────────────────────────────────────
def load_image(path: str) -> Image.Image:
    """RGB 모드로 이미지를 불러온다."""
    return Image.open(path).convert("RGB")


def resize_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    PIL 이미지를 주어진 (W, H) size로 **BILINEAR** 리샘플링해 리턴.

    모델 학습·PC 측 전처리와 동일하게 맞추기 위해 변경
    (기존 LANCZOS → BILINEAR).
    """
    return img.resize(size, resample=Image.BILINEAR)


# ────────────────────────────────────────────────────────────
# 모델 입출력 보조
# ────────────────────────────────────────────────────────────
def preprocess_for_model(img: Image.Image, in_size: Tuple[int, int]) -> Image.Image:
    """
    모델 입력 사이즈에 맞춰 리사이즈한 뒤 그대로 PIL.Image를 반환한다.
    (Edge TPU API가 PIL 객체를 직접 받으므로 추가 변환 없음)
    """
    return resize_image(img, in_size)


def tensor_to_mask(output: np.ndarray) -> np.ndarray:
    """
    TPU 추론 결과 텐서를 2D 클래스 인덱스 맵으로 변환.
    - (H, W) : 바로 반환
    - (H, W, C) : argmax 채널 축
    - 그 외 shape → ValueError
    """
    if output.ndim == 2:
        return output.astype(np.uint8)
    if output.ndim == 3:
        return np.argmax(output, axis=-1).astype(np.uint8)
    raise ValueError(f"Unsupported mask shape: {output.shape}")


# ────────────────────────────────────────────────────────────
# 마스크 후처리
# ────────────────────────────────────────────────────────────
def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    (h,w) 혹은 (h,w,…) 마스크를 원본 (W,H) 사이즈로 NEAREST 리샘플링.
    """
    return np.array(
        Image.fromarray(mask.astype(np.uint8)).resize(size, resample=Image.NEAREST)
    )


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """팔레트( N×3 uint8)로 색칠된 (H,W,3) 배열 반환."""
    return palette[mask]


def blend_mask(
    orig: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    컬러 마스크를 원본과 블렌딩.
    - mask==0(배경) 영역은 그대로 둔다.
    - alpha : 마스크 쪽 불투명도
    """
    colored = colorize_mask(mask, palette)
    out = orig.copy()
    region = mask != 0
    out[region] = (
        out[region] * (1 - alpha) + colored[region] * alpha
    ).astype(np.uint8)
    return out
