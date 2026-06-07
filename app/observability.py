"""Observability — structured logging, request correlation, optional Sentry.

Installs the logging config (JSON/CLI + ring buffer), assigns a correlation id to
every request, logs request/response with timing + status, and reports errors to
Sentry when configured. Per-stage pipeline timings come from the Modal worker.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, Request

from app.logging_config import configure_logging, request_id_var

logger = logging.getLogger("pvllm.api")


def init_observability(app: FastAPI) -> None:
    configure_logging()

    dsn = os.getenv("SENTRY_DSN")
    if dsn:
        try:
            import sentry_sdk  # lazy, optional
            sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1,
                            environment=os.getenv("ENVIRONMENT", "dev"))
            logger.info("sentry initialised")
        except ImportError:
            logger.warning("SENTRY_DSN set but sentry_sdk not installed")

    @app.middleware("http")
    async def _correlate(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request failed", extra={
                "method": request.method, "path": request.url.path})
            request_id_var.reset(token)
            raise
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        level = logging.WARNING if response.status_code >= 500 else logging.INFO
        logger.log(level, f"{request.method} {request.url.path}", extra={
            "status": response.status_code, "ms": elapsed_ms,
            "method": request.method, "path": request.url.path})
        response.headers["X-Request-ID"] = rid
        response.headers["X-Response-Time-ms"] = str(elapsed_ms)
        request_id_var.reset(token)
        return response
