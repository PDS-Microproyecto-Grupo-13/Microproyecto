import contextvars
import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Context variable to hold request_id across async execution
_request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> str | None:
    """Returns the current request ID from context if available."""
    return _request_id_ctx_var.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for injecting Request-ID, logging request lifecycle, and measuring latency."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        # Extract or generate X-Request-ID
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = _request_id_ctx_var.set(request_id)

        start_time = time.perf_counter()

        logger.info(
            "request_started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            # Import dynamically to avoid circular dependencies
            from app.core.exceptions import unhandled_exception_handler

            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
                exc_info=True,
            )
            response = await unhandled_exception_handler(request, exc)
        finally:
            _request_id_ctx_var.reset(token)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Attach header to outgoing response
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
