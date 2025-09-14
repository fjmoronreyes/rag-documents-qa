import logging
import sys
from typing import Dict


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return f"{self.COLORS.get(record.levelname, self.RESET)}{msg}{self.RESET}"


LEVEL_MAPPER: Dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def get_log_level(level: str) -> int:
    return LEVEL_MAPPER.get(level.lower(), logging.INFO)


def get_logger(name: str = "RAG", level: str = "info") -> logging.Logger:
    """
    Creates and configures a colored console logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level(level))

    if not logger.hasHandlers():
        fmt = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(get_log_level(level))
        console_handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))

        logger.addHandler(console_handler)

    return logger
