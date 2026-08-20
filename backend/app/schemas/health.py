from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Healthcheck endpoint response schema."""

    status: str = Field(default="ok", description="Service operational status")
    service: str = Field(..., description="Service application name")
    version: str = Field(..., description="Semantic version of the deployed application")
