import json
import logging
from fraudguard.monitoring.logger import get_logger, JsonFormatter


class TestJsonFormatter:

    def test_format_produces_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_format_includes_exception(self) -> None:
        formatter = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR,
            pathname="", lineno=0,
            msg="error occurred", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestGetLogger:

    def test_returns_logger_instance(self) -> None:
        logger = get_logger("fraudguard.test")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self) -> None:
        logger = get_logger("fraudguard.test.handler")
        assert len(logger.handlers) >= 1

    def test_same_name_returns_same_logger(self) -> None:
        logger_a = get_logger("fraudguard.singleton")
        logger_b = get_logger("fraudguard.singleton")
        assert logger_a is logger_b
