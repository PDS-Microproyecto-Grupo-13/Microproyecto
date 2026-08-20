from fastapi import APIRouter, Depends, status

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the operational status, application name, and deployed version.",
)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Endpoint to verify application liveness and status."""
    return HealthService.get_health_status(settings=settings)
