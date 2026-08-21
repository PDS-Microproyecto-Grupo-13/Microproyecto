from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standardized API error response contract."""

    error: str = Field(..., description="Machine-readable error identifier code")
    message: str = Field(..., description="Human-readable summary of the error")
    request_id: str | None = Field(
        default=None, description="Unique trace identifier for the request"
    )
    details: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Optional detailed error context or validation errors"
    )
