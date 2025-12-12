"""Structured logging utilities for training and evaluation.

Provides consistent logging across all modules with proper formatting,
log levels, and file/console output.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import torch


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output.

    Adds ANSI color codes to log levels for better readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format.

        Returns:
            Formatted log string with ANSI colors.
        """
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    use_colors: bool = True,
) -> logging.Logger:
    """Setup logger with file and console handlers.

    Args:
        name: Logger name.
        log_file: Optional log file path.
        level: Logging level.
        use_colors: Use colored output for console.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers.clear()

    # Format string
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors:
        console_formatter = ColoredFormatter(fmt, datefmt=datefmt)
    else:
        console_formatter = logging.Formatter(fmt, datefmt=datefmt)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log system and environment information.

    Args:
        logger: Logger to use for output.
    """
    import platform

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("=" * 60)
