import contextvars
import logging
import time
import uuid

from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Context variable to hold request_id across async execution
_request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def get_request_id() -> str | None:
    """Returns the current request ID from context if available."""
    return _request_id_ctx_var.get()


class RequestContextMiddleware:
    """Middleware for injecting Request-ID, logging request lifecycle, and measuring latency."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate X-Request-ID
        request = Request(scope, receive=receive)
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = _request_id_ctx_var.set(request_id)
        start_time = time.perf_counter()
        status_code = 500
        response_started = False

        async def send_with_context(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = [
                    header
                    for header in message.get("headers", [])
                    if header[0].lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        logger.info(
            "request_started",
            extra={
                "request_id": request_id,
                "method": scope["method"],
                "path": scope["path"],
                "client": (scope.get("client") or (None, None))[0],
            },
        )

        try:
            await self.app(scope, receive, send_with_context)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
                exc_info=True,
            )
            if response_started:
                raise

            # Import dynamically to avoid circular dependencies
            from app.core.exceptions import unhandled_exception_handler

            response = await unhandled_exception_handler(request, exc)
            await response(scope, receive, send_with_context)
        finally:
            _request_id_ctx_var.reset(token)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": scope["method"],
                "path": scope["path"],
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
