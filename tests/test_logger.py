"""Tests for the logging setup module."""

import logging

from src.utils.logger import get_logger, setup_logging


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_creates_handler(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()

        setup_logging("DEBUG")
        assert len(root.handlers) >= 1
        assert root.level == logging.DEBUG

        # Cleanup
        root.handlers.clear()

    def test_setup_idempotent(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()

        setup_logging("INFO")
        count = len(root.handlers)
        setup_logging("INFO")
        assert len(root.handlers) == count

        root.handlers.clear()

    def test_setup_invalid_level_defaults_to_info(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()

        setup_logging("NONEXISTENT")
        assert root.level == logging.INFO

        root.handlers.clear()


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_named_logger(self) -> None:
        logger = get_logger("test.module")
        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_same_name_returns_same_logger(self) -> None:
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2
