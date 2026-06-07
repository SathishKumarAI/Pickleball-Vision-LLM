"""Observability — structured request logging + optional Sentry.

Env-gated and lazy: no hard dependency on Sentry. If ``SENTRY_DSN`` is set and
``sentry_sdk`` is installed, errors are reported; otherwise this is a no-op plus
lightweight request timing logs. Per-stage pipeline timings are emitted by the
Modal worker into the job row (see worker/runner).
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, Request

logger = logging.getLogger("pvllm.api")


def init_observability(app: FastAPI) -> None:
    """Attach request-timing middleware and (optionally) Sentry."""
    dsn = os.getenv("SENTRY_DSN")
    if dsn:
        try:
            import sentry_sdk  # lazy, optional
            sentry_sdk.init(dsn=dsn, traces_sample_rate=0.1,
                            environment=os.getenv("ENVIRONMENT", "dev"))
            logger.info("Sentry initialised")
        except ImportError:
            logger.warning("SENTRY_DSN set but sentry_sdk not installed")

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("%s %s -> %s %.1fms",
                    request.method, request.url.path, response.status_code, elapsed_ms)
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        return response
