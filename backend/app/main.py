import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.middleware.request_context import RequestContextMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context managing startup and shutdown lifecycle events."""
    settings = get_settings()

    logger.info(
        "application_started",
        extra={
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "log_level": settings.LOG_LEVEL,
            "log_format": settings.LOG_FORMAT,
        },
    )

    yield

    logger.info(
        "application_stopped",
        extra={
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory configuring logging, middleware, routers, and handlers."""
    if settings is None:
        settings = get_settings()

    # Initialize centralized logging
    setup_logging(settings)

    # Instantiate FastAPI app with OpenAPI docs
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    cors_origins = (
        settings.CORS_ORIGINS
        if isinstance(settings.CORS_ORIGINS, list)
        else [settings.CORS_ORIGINS]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Request Context / Tracing Middleware
    app.add_middleware(RequestContextMiddleware)

    # Register Centralized Exception Handlers
    register_exception_handlers(app)

    # Include Versioned API Router
    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


# Root ASGI application instance
app = create_app()
