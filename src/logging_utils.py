"""Shared logging helpers for the study pipeline."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger once and return the study logger."""
    logger = logging.getLogger("btc_fng_study")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_count(logger: logging.Logger, label: str, n: int) -> None:
    """Log a row/count milestone."""
    logger.info("%s: %d", label, n)
