from app.core.config import Settings
from app.schemas.health import HealthResponse


class HealthService:
    """Service providing application health status information."""

    @staticmethod
    def get_health_status(settings: Settings) -> HealthResponse:
        """Returns the current operational status of the service."""
        return HealthResponse(
            status="ok",
            service=settings.APP_NAME,
            version=settings.APP_VERSION,
        )
