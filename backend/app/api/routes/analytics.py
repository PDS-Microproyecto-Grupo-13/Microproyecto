from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, Header, Response, status

from app.core.config import Settings, get_settings
from app.core.exceptions import ApplicationError
from app.schemas.analytics import AnalyticsPublishResponse, AnalyticsSummary
from app.schemas.common import ErrorResponse
from app.services.analytics_service import AnalyticsStorageService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


async def get_analytics_storage_service(
    settings: Settings = Depends(get_settings),
) -> AnalyticsStorageService:
    return AnalyticsStorageService(Path(settings.ANALYTICS_STORAGE_PATH))


async def require_analytics_publish_token(
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
) -> None:
    configured_secret = settings.ANALYTICS_PUBLISH_TOKEN
    if configured_secret is None or not configured_secret.get_secret_value():
        raise ApplicationError(
            message="Analytics publication is not configured",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="analytics_publish_unavailable",
        )

    scheme, separator, provided_token = (authorization or "").partition(" ")
    expected_token = configured_secret.get_secret_value()
    token_matches = separator == " " and scheme == "Bearer" and bool(provided_token)
    if token_matches:
        token_matches = secrets.compare_digest(
            provided_token.encode("utf-8"), expected_token.encode("utf-8")
        )
    if not token_matches:
        raise ApplicationError(
            message="Invalid analytics publication credentials",
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="analytics_unauthorized",
        )


@router.post(
    "/snapshots",
    response_model=AnalyticsPublishResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def publish_analytics_snapshot(
    summary: AnalyticsSummary,
    response: Response,
    _: None = Depends(require_analytics_publish_token),
    storage: AnalyticsStorageService = Depends(get_analytics_storage_service),
) -> AnalyticsPublishResponse:
    result = storage.publish(summary)
    response.status_code = (
        status.HTTP_201_CREATED if result.status == "created" else status.HTTP_200_OK
    )
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get(
    "/summary",
    response_model=AnalyticsSummary,
    responses={
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_analytics_summary(
    response: Response,
    storage: AnalyticsStorageService = Depends(get_analytics_storage_service),
) -> AnalyticsSummary:
    response.headers["Cache-Control"] = "no-store"
    return storage.get_summary()
