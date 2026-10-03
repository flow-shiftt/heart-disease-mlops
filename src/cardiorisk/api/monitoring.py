"""Structured request logging and Prometheus metrics for the API."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "cardiorisk_http_requests_total", "HTTP requests by route and status",
    ["method", "route", "status"],
)
LATENCY = Histogram(
    "cardiorisk_http_request_duration_seconds", "Request latency by route", ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter(
    "cardiorisk_predictions_total", "Predictions served by outcome and risk band",
    ["label", "risk_band"],
)
PROBABILITY = Histogram(
    "cardiorisk_prediction_probability", "Distribution of predicted disease probability "
    "(a shift here is an early sign of input drift)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
VALIDATION_ERRORS = Counter(
    "cardiorisk_validation_errors_total", "Rejected payloads by offending field", ["field"],
)
MODEL_INFO = Gauge(
    "cardiorisk_model_info", "Currently loaded model (value is always 1)", ["name", "version"],
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line so logs are greppable and ingestible by Loki/ELK."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class _StdoutHandler(logging.StreamHandler):
    """Resolves sys.stdout at emit time (works with uvicorn reloads and pytest capture)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("cardiorisk.api")
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    handlers: list[logging.Handler] = [_StdoutHandler()]
    if log_file := os.getenv("LOG_FILE"):
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
