"""Validation for raw Foorilla records before they're written to CSV.

Field names for /hiring/job/ confirmed 2026-08-15 via a manual CSV export
from the Foorilla dashboard — see JobRaw below. /insight/salary/'s schema
is still unconfirmed (see SalaryRaw and README).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobRaw(BaseModel):
    """Validated record from /hiring/job/. Field set confirmed against a
    manual CSV export from Foorilla's dashboard (2026-08-15)."""
    model_config = ConfigDict(extra="allow")  # keep any fields Foorilla adds later

    company: str | None = None
    title: str | None = None
    location: str | None = None
    has_remote: bool | None = None
    is_agency: bool | None = None
    published: str | None = None
    expired: str | None = None
    experience_level: str | None = None
    experience_years: float | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    views: int | None = None
    clicks: int | None = None
    foo_url: str | None = None
    apply_url: str | None = None

    @classmethod
    def from_api_record(cls, record: dict[str, Any]) -> "JobRaw":
        return cls(**record)


class SalaryRaw(BaseModel):
    """Loose validation for records from /insight/salary/.
    Field names are placeholders pending confirmation — this endpoint's
    schema hasn't been confirmed the way /hiring/job/ has (see JobRaw above).
    """
    model_config = ConfigDict(extra="allow")

    id: Any = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api_record(cls, record: dict[str, Any]) -> "SalaryRaw":
        return cls(id=record.get("id"), raw=record, **{k: v for k, v in record.items() if k != "id"})
