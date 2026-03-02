"""
logging.py — Structured logging configuration using structlog.

Structlog outputs machine-readable JSON in production and
human-readable colored output in development (TTY detection).
"""

import logging
import sys

import structlog
from rich.logging import RichHandler


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog + standard library logging.
    - TTY (your terminal): rich colored output
    - Non-TTY (CI, Docker, files): JSON output
    """
    is_tty = sys.stderr.isatty()

    # Configure standard lib logging to use Rich in TTY
    logging.basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_time=False,
                show_path=False,
            )
            if is_tty
            else logging.StreamHandler()
        ],
    )

    # Shared processors for all structlog calls
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_tty:
        # Human-readable colored output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Machine-readable JSON for production / CI / Docker
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """Get a structlog logger bound to the given name."""
    return structlog.get_logger(name)
