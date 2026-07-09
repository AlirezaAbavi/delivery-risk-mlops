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

# A ContextVar is like a thread-local, but it also works correctly under async: each
# request handled by the event loop sees its own value. The middleware ``set()``s the
# incoming request id here; every log line emitted while that request is being served
# reads it back, so all lines for one request share a correlation id. "-" is the value
# outside any request (e.g. startup logs).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Both knobs are env-driven so ops can raise verbosity or switch to plain text without
# a code change (12-factor config).
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()

# The trick for "structured logging" with stdlib logging: callers attach fields via
# ``log.info("event", extra={"order_id": ...})``. Those become attributes on the
# LogRecord. To emit *only* the caller's fields (not Python's ~20 built-in record
# attributes) we snapshot the built-in attribute names of a blank record here, then
# subtract them when serialising. ``taskName`` etc. are added for newer Pythons.
_STD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Serialise each log record as a single JSON line (one event per line).

    JSON-per-line is what log collectors (Promtail -> Loki -> Grafana) parse best: the
    standard fields are always present, and any ``extra=`` fields ride along as
    first-class keys you can filter/aggregate on downstream.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Start with the always-present envelope fields.
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),  # correlation id for this request
        }
        # Fold in every caller-supplied ``extra=`` field (anything not a built-in
        # LogRecord attribute and not a private ``_`` key) as a top-level JSON key.
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        # If the log call carried an exception (e.g. log.exception / exc_info=True),
        # attach the formatted traceback as a field instead of losing it.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # ``default=str`` guarantees serialisation never fails on an odd value type.
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install a single stdout handler on the root logger (idempotent).

    We log to stdout (not a file) on purpose: in a container the platform captures
    stdout, so the app shouldn't manage log files itself. Called once at import time
    in main.py, before anything logs. Idempotent because it clears existing handlers
    first — safe if invoked twice (e.g. by the test client).
    """
    handler = logging.StreamHandler(sys.stdout)
    # Plain text is handy for eyeballing locally; JSON is the default for ingestion.
    if LOG_FORMAT == "text":
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())

    # Reset the root logger to exactly our one handler at the configured level.
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)

    # Uvicorn installs its own handlers by default, which would double-log and bypass
    # our JSON format. Clear them and let those records propagate up to the root logger
    # so *all* logs — app and server — come out in one consistent format.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
