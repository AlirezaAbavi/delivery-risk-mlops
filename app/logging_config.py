"""Structured JSON logging for the delivery-risk API.

Logs go to stdout as one JSON object per line so a collector (Promtail/Docker ->
Loki -> Grafana) can ingest them. A ``request_id`` contextvar correlates every log
line emitted while handling a single request. Configurable via env:
  LOG_LEVEL  (default INFO)
  LOG_FORMAT (json | text, default json)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Set per-request by the logging middleware; "-" outside a request scope.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()

# Standard LogRecord attributes we don't want to duplicate into the JSON body;
# anything passed via ``extra=`` will not be in this set and is emitted as a field.
_STD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install the stdout handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    if LOG_FORMAT == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)

    # Route uvicorn's loggers through the same handler/format.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
