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
    """Matched route pattern (stable label); avoids per-path cardinality blowups."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return "<unmatched>"


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        method = request.method
        started = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
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

            duration = time.perf_counter() - started
            endpoint = _endpoint(request)
            status = str(response.status_code)
            metrics.HTTP_REQUESTS.labels(method, endpoint, status).inc()
            metrics.HTTP_LATENCY.labels(endpoint).observe(duration)
            if response.status_code >= 400:
                metrics.HTTP_ERRORS.labels(endpoint, status).inc()

            # Keep scrape traffic quiet; warn on server errors.
            level = logging.DEBUG if endpoint == "/metrics" else logging.INFO
            if response.status_code >= 500:
                level = logging.WARNING
            log.log(
                level, "request",
                extra={"method": method, "endpoint": endpoint,
                       "status": response.status_code, "latency_ms": round(duration * 1000, 2)},
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)
