"""
파일·경로 관련 유틸리티
Author : 상현님 프로젝트
"""

from __future__ import annotations

import json
import os
from typing import Generator, Dict, Any, List


def load_coco_annotations(path: str) -> Dict[str, Any]:
    """COCO 형식 JSON 로드."""
    with open(path, "r") as f:
        return json.load(f)


def image_infos_generator(
    coco_dict: Dict[str, Any],
    image_root: str,
    exists_only: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    COCO images 목록을 순회하며 info dict를 yield.

    Parameters
    ----------
    exists_only : True면 실제 파일이 존재할 때만 yield.
    """
    for info in coco_dict.get("images", []):
        if not exists_only:
            yield info
        else:
            fp = os.path.join(image_root, info["file_name"])
            if os.path.isfile(fp):
                yield info


def missing_images(
    coco_dict: Dict[str, Any], image_root: str
) -> List[str]:
    """COCO에 등록됐지만 존재하지 않는 이미지 파일 경로 리스트."""
    return [
        os.path.join(image_root, info["file_name"])
        for info in coco_dict.get("images", [])
        if not os.path.isfile(os.path.join(image_root, info["file_name"]))
    ]
