"""Structured logging configuration for Stockpile.

Uses structlog for structured, JSON-capable logging with job correlation.
"""

import logging
import sys
from contextvars import ContextVar

import structlog

# Context variable for job ID correlation
current_job_id: ContextVar[str | None] = ContextVar("current_job_id", default=None)


def add_job_id(_logger, _method_name, event_dict):
    """Structlog processor to inject job_id into all log events."""
    job_id = current_job_id.get()
    if job_id:
        event_dict["job_id"] = job_id
    return event_dict


def setup_logging(log_level: str = "INFO", json_output: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, output JSON logs (for production). If False, use colored console output.
    """
    # Shared processors for both structlog and stdlib
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_job_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for production/log aggregation
        renderer = structlog.processors.JSONRenderer()
    else:
        # Colored console output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Suppress noisy third-party loggers
    noisy_loggers = [
        "httpx",
        "google_genai",
        "google_genai.models",
        "googleapiclient.discovery_cache",
        "google_auth_oauthlib.flow",
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "aiosqlite",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Bound structlog logger
    """
    return structlog.get_logger(name)


def set_job_context(job_id: str) -> None:
    """Set the current job ID for log correlation.

    Args:
        job_id: Job ID to include in all subsequent log messages
    """
    current_job_id.set(job_id)


def clear_job_context() -> None:
    """Clear the current job context."""
    current_job_id.set(None)
