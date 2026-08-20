import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.middleware.request_context import get_request_id
from app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    """Base exception for all application-level errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "application_error",
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


class ExternalServiceError(ApplicationError):
    """Exception raised when an upstream external service (e.g. MLflow / inference) fails."""

    def __init__(
        self,
        service_name: str,
        message: str,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(
            message=f"External service '{service_name}' error: {message}",
            status_code=status_code,
            error_code="external_service_error",
            details=details or {"service": service_name},
        )
        self.service_name = service_name


async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
    """Handler for handled ApplicationError exceptions."""
    request_id = get_request_id()
    logger.warning(
        "application_error",
        extra={
            "request_id": request_id,
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "error_detail": exc.message,
            "path": request.url.path,
        },
    )
    content = ErrorResponse(
        error=exc.error_code,
        message=exc.message,
        request_id=request_id,
        details=exc.details if exc.details else None,
    ).model_dump(exclude_none=True)
    return JSONResponse(status_code=exc.status_code, content=content)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for standard FastAPI/Starlette HTTPExceptions."""
    request_id = get_request_id()
    logger.warning(
        "http_exception",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )
    content = ErrorResponse(
        error="http_error",
        message=str(exc.detail),
        request_id=request_id,
    ).model_dump(exclude_none=True)
    return JSONResponse(status_code=exc.status_code, content=content)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handler for schema validation errors."""
    request_id = get_request_id()
    logger.warning(
        "validation_error",
        extra={
            "request_id": request_id,
            "errors": exc.errors(),
            "path": request.url.path,
        },
    )
    content = ErrorResponse(
        error="validation_error",
        message="Request payload validation failed",
        request_id=request_id,
        details=exc.errors(),
    ).model_dump(exclude_none=True)
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=content)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled exceptions to prevent stack trace leakage."""
    request_id = get_request_id()
    logger.error(
        "unexpected_error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error_type": exc.__class__.__name__,
        },
        exc_info=True,
    )
    content = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred",
        request_id=request_id,
    ).model_dump(exclude_none=True)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content)


def register_exception_handlers(app: FastAPI) -> None:
    """Registers centralized exception handlers into the FastAPI application instance."""
    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
