"""Structured logging configuration for Sproxxo MLOps platform."""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.stdlib import LoggerFactory

from ..config import settings


def setup_logging() -> None:
    """Structured logging with structlog."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Create logger instance
    logger = structlog.get_logger()
    logger.info("Logging configured", log_level=settings.log_level, environment=settings.environment)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name

    Returns
    -------
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_prediction(
    logger: structlog.BoundLogger,
    transaction_id: str,
    fraud_probability: float,
    is_fraud: bool,
    model_version: str,
    processing_time_ms: float,
    **kwargs: Any,
) -> None:
    """Log a fraud prediction with structured data.

    Args:
        logger: Logger instance
        transaction_id: Transaction identifier
        fraud_probability: Predicted fraud probability
        is_fraud: Binary fraud prediction
        model_version: Model version used
        processing_time_ms: Processing time in milliseconds
        **kwargs: Additional fields to log
    """
    logger.info(
        "Fraud prediction made",
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
        **kwargs,
    )


def log_model_event(logger: structlog.BoundLogger, event_type: str, model_version: str, **kwargs: Any) -> None:
    """Log model-related events.

    Args:
        logger: Logger instance
        event_type: Type of model event
        model_version: Model version
        **kwargs: Additional fields to log
    """
    logger.info("Model event", event_type=event_type, model_version=model_version, **kwargs)


def log_error(
    logger: structlog.BoundLogger, error: Exception, context: Optional[dict[str, Any]] = None, **kwargs: Any
) -> None:
    """Log errors with structured context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
        **kwargs: Additional fields to log
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_traceback": getattr(error, "__traceback__", None),
    }

    if context:
        error_data.update(context)

    logger.error("Error occurred", **error_data, **kwargs)
