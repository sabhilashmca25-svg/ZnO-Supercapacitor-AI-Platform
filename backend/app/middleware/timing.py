"""
timing.py — Request timing + structured access log middleware.

Logs every HTTP request with:
  METHOD  path  →  status  in Xs.XXXs

Also adds an X-Response-Time header to every response so the frontend
and browser DevTools can show latency without inspecting logs.
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("access")


class TimingMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that times every request and emits a structured access log.

    Skips noisy health/readiness probes from the access log to keep logs clean.
    """

    # Endpoints polled by load-balancers / uptime monitors — suppress from logs
    _SILENT_PATHS = {"/api/v1/health", "/api/v1/ready"}

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error(
                "UNHANDLED  %s %s  →  500  %.3fs",
                request.method,
                request.url.path,
                elapsed,
            )
            raise

        elapsed = time.perf_counter() - start
        ms_str  = f"{elapsed:.3f}s"

        # Emit access log (skip health-check noise)
        if request.url.path not in self._SILENT_PATHS:
            logger.info(
                "%s  %s  →  %s  %s",
                request.method.ljust(6),
                request.url.path,
                response.status_code,
                ms_str,
            )

        # Expose latency via response header (visible in browser DevTools)
        response.headers["X-Response-Time"] = ms_str
        return response
