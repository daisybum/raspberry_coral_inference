"""
Matplotlib 시각화 모음
Author : 상현님 프로젝트
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def create_legend_patches(
    palette: np.ndarray | List[Tuple[int, int, int]],
    class_names: List[str],
):
    """팔레트·클래스명으로 matplotlib Patch 리스트 생성."""
    patches: List[mpatches.Patch] = []
    for idx, (r, g, b) in enumerate(palette):
        patches.append(
            mpatches.Patch(
                color=(r / 255, g / 255, b / 255),
                label=class_names[idx],
            )
        )
    return patches


def visualize_and_save(
    file_name: str,
    orig: np.ndarray,
    color_mask: np.ndarray,
    overlay: np.ndarray,
    legend_patches: List[mpatches.Patch],
    output_dir: str,
    save_image: bool = True,
) -> str:
    """
    원본/마스크/오버레이 3분할 이미지 저장 후 경로 문자열 반환.
    
    Parameters:
        file_name: Name of the file being processed
        orig: Original image as numpy array
        color_mask: Colorized mask as numpy array
        overlay: Overlay image as numpy array
        legend_patches: Legend patches for the visualization
        output_dir: Directory to save the output image
        save_image: If True, save the visualization to a file; if False, just create the figure without saving
    
    Returns:
        Path to the saved image or a placeholder string if save_image is False
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Segmentation – {file_name}", fontsize=16)

    titles = ["Original", "Mask", "Overlay"]
    images = [orig, color_mask, overlay]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=12,
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    out_path = os.path.join(
        output_dir, f"{os.path.splitext(file_name)[0]}_visual.png"
    )
    
    if save_image:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path)
    
    plt.close(fig)
    return out_path if save_image else "[Visualization created but not saved]"
