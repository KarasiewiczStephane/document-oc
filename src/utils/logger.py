"""Centralized logging setup for the document OCR system.

Provides a structured logging configuration with consistent formatting
across all modules.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a standard format.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
