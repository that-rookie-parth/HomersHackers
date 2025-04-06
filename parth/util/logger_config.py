# logger_config.py

import logging
from colorlog import ColoredFormatter

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Creates and returns a colorized logger with the given name and level (without timestamp)."""

    # Log format without timestamp
    log_format = "%(log_color)s[%(levelname)s] %(message)s"

    formatter = ColoredFormatter(
        log_format,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers
    if not logger.hasHandlers():
        logger.addHandler(stream)

    return logger
