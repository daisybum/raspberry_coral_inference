"""
간단 성능 계측용 contextmanager
Author : daisybum
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

from .logger import get_logger


@contextmanager
def elapsed(msg: str, logger_name: str = "timer"):
    """
    with elapsed("step 설명"): 블록 실행 시간을 로깅한다.
    """
    logger = get_logger(logger_name)
    t0 = time.time()
    yield
    dt = time.time() - t0
    logger.info(f"{msg} – {dt:.3f}s")
