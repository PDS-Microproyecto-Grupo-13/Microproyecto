import datetime
import json
import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from app.core.config import Settings


class JSONLogFormatter(logging.Formatter):
    """Structured JSON log formatter for production Docker stdout."""

    def format(self, record: logging.LogRecord) -> str:
        # Import dynamically to avoid circular import issues
        from app.middleware.request_context import get_request_id

        log_data: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Resolve request_id from record or context variable
        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        # Add structured extra attributes if present
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_") and key != "request_id":
                log_data[key] = value

        # Handle exception tracebacks if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def setup_logging(settings: Settings) -> None:
    """Configures centralized logging according to application settings."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate log entries
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if settings.LOG_FORMAT == "json":
        json_handler = logging.StreamHandler(sys.stdout)
        json_handler.setFormatter(JSONLogFormatter())
        root_logger.addHandler(json_handler)
    else:
        console = Console(file=sys.stdout, color_system="auto")
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        # Custom format that appends key-value context cleanly
        rich_formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
        rich_handler.setFormatter(rich_formatter)
        root_logger.addHandler(rich_handler)

    # Configure uvicorn loggers to avoid redundant lines while preserving levels
    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    # Disable default access logger if request_context middleware is handling request logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
