from __future__ import annotations

import logging

_logger_name = "twrapform"


def get_logger(loglevel=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(_logger_name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(loglevel)
        logger.addHandler(handler)
    return logger
