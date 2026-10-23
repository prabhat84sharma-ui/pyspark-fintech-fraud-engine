"""Enterprise structured logger utility for pyspark-fintech-fraud-engine."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter producing consistent, structured log output."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(
            fmt=fmt or "%(asctime)s | %(levelname)-7s | [%(name)s:%(lineno)d] - %(message)s",
            datefmt=datefmt or "%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        # Support optional structured context attributes
        context_parts = []
        for attr in ("batch_id", "stage", "record_count", "elapsed_sec"):
            if hasattr(record, attr):
                context_parts.append(f"{attr}={getattr(record, attr)}")

        formatted = super().format(record)
        if context_parts:
            formatted = f"{formatted} | {', '.join(context_parts)}"
        return formatted


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """Creates and returns a configured logger instance.

    Args:
        name: Name of the logger, typically __name__ of the caller module.
        log_level: Optional log level string (DEBUG, INFO, WARN, ERROR). Defaults
            to LOG_LEVEL environment variable or INFO.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    level_name = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate logs in Spark driver
    logger.propagate = False

    return logger
