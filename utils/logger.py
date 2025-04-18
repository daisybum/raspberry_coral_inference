"""
프로젝트 공통 로거 헬퍼
Author : 상현님 프로젝트
"""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    stream=None,
) -> logging.Logger:
    """
    일관된 포맷의 Logger 반환. (중복 핸들러 방지)
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # 이미 설정된 경우
        return logger

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
