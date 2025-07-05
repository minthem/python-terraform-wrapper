import logging

from twrapform.logging import get_logger


def test_get_logger_info_level(caplog):
    logger = get_logger(logging.INFO)

    with caplog.at_level(logging.INFO):
        logger.info("test output")

    assert "test output" in caplog.text
    assert "INFO" in caplog.text
    assert "twrapform" in caplog.text


def test_logger_is_singleton():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2
