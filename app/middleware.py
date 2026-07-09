"""Request-logging + HTTP-metrics middleware.

Assigns/propagates an ``X-Request-ID``, times every request, emits a structured
access log, records HTTP-level Prometheus metrics, and converts any unhandled
exception into a structured 500 (logged with traceback) so the service never
leaks a stack trace to the client.
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import metrics
from .logging_config import request_id_var

log = logging.getLogger("api.access")


def _endpoint(request: Request) -> str:
    """Return the matched *route pattern* (e.g. "/predict"), not the raw URL path.

    This is a critical Prometheus-cardinality guard. If we labelled metrics by the raw
    path, an endpoint like ``/orders/{id}`` would spawn a brand-new time series for
    every distinct id — potentially millions — and blow up Prometheus's memory. The
    route pattern collapses all of those into one stable label. ``<unmatched>`` covers
    requests to paths with no route (404s), which likewise mustn't create per-URL series.
    """
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return "<unmatched>"


class LoggingMiddleware(BaseHTTPMiddleware):
    """Wraps every request to add correlation ids, structured access logs, HTTP-level
    Prometheus metrics, and a safety net that converts any unhandled exception into a
    clean structured 500. Being middleware, it sees *all* requests uniformly — the
    endpoint handlers stay free of cross-cutting boilerplate.
    """

    async def dispatch(self, request: Request, call_next):
        # Reuse an inbound X-Request-ID if the caller/proxy set one (so a trace id can
        # span multiple services); otherwise mint a fresh one. Store it in the
        # contextvar so every log line during this request is correlated; ``token``
        # lets us restore the previous value afterwards.
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        method = request.method
        started = time.perf_counter()  # monotonic clock for measuring duration
        try:
            # Inner try: run the actual handler. If it raises, we own the failure.
            try:
                response = await call_next(request)
            except Exception:
                # An exception escaped the handler. Record it as a 500 in the metrics,
                # log the full traceback, and return a *sanitised* JSON body — the
                # client gets a request id to quote, never a raw stack trace (which
                # could leak internals). The service stays up.
                duration = time.perf_counter() - started
                endpoint = _endpoint(request)
                metrics.HTTP_REQUESTS.labels(method, endpoint, "500").inc()
                metrics.HTTP_ERRORS.labels(endpoint, "500").inc()
                metrics.HTTP_LATENCY.labels(endpoint).observe(duration)
                log.exception(
                    "unhandled_exception",
                    extra={"method": method, "endpoint": endpoint,
                           "latency_ms": round(duration * 1000, 2)},
                )
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error", "request_id": request_id},
                    headers={"X-Request-ID": request_id},
                )

            # Normal path: the handler returned. Record request count + latency, and
            # count it as an error too if it's a 4xx/5xx status.
            duration = time.perf_counter() - started
            endpoint = _endpoint(request)
            status = str(response.status_code)
            metrics.HTTP_REQUESTS.labels(method, endpoint, status).inc()
            metrics.HTTP_LATENCY.labels(endpoint).observe(duration)
            if response.status_code >= 400:
                metrics.HTTP_ERRORS.labels(endpoint, status).inc()

            # Pick a log level that keeps the signal useful: Prometheus scrapes /metrics
            # constantly, so log those at DEBUG to avoid drowning the logs; normal
            # requests at INFO; server errors at WARNING so they stand out.
            level = logging.DEBUG if endpoint == "/metrics" else logging.INFO
            if response.status_code >= 500:
                level = logging.WARNING
            log.log(
                level, "request",
                extra={"method": method, "endpoint": endpoint,
                       "status": response.status_code, "latency_ms": round(duration * 1000, 2)},
            )
            # Echo the correlation id back so the caller can quote it in a bug report.
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # Always restore the contextvar, even on error, so ids never leak between
            # requests that reuse the same worker/task.
            request_id_var.reset(token)
