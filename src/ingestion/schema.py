"""Validation for raw Foorilla records before they're written to CSV.

"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TagRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int | None = None
    name: str | None = None
    tag_type: str | None = None


class TopicRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int | None = None
    name: str | None = None
    is_main: bool | None = None


class NamedRef(BaseModel):
    """Generic {id, name} shape used for regions and similar simple refs."""
    model_config = ConfigDict(extra="allow")
    id: int | None = None
    name: str | None = None


class CountryRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int | None = None
    name: str | None = None
    code: str | None = None


class CompanyRef(BaseModel):
    """Nested company object. Only the fields likely useful for salary
    modeling/joins are modeled explicitly; extra="allow" keeps the rest
    (website, socials, hq_city/hq_state/hq_country geo detail, founded, etc.)
    available on `.model_extra` without needing every one named here."""
    model_config = ConfigDict(extra="allow")

    id: int | None = None
    name: str | None = None
    hq: str | None = None
    size: int | None = None
    is_agency: bool | None = None
    is_remote: bool | None = None


class JobRaw(BaseModel):
    """Validated record from /hiring/job/, confirmed against the real
    OpenAPI schema (2026-08-15)."""
    model_config = ConfigDict(extra="allow")  # keep any fields Foorilla adds later

    id: int | None = None
    company: CompanyRef | str | None = None
    title: str | None = None
    location: str | None = None
    published: str | None = None
    expired: str | None = None
    experience_level: str | None = None
    experience_years: float | None = None
    language: str | None = None
    apply_url: str | None = None

    salary_min: float | None = None
    salary_max: float | None = None
    salary_min_est: float | None = None
    salary_max_est: float | None = None
    salary_currency: str | None = None
    salary_min_usd: float | None = None
    salary_max_usd: float | None = None
    salary_min_eur: float | None = None
    salary_max_eur: float | None = None

    has_remote: bool | None = None
    work_mode: int | None = None

    regions: list[NamedRef] = Field(default_factory=list)
    countries: list[CountryRef] = Field(default_factory=list)
    topics: list[TopicRef] = Field(default_factory=list)
    tags: list[TagRef] = Field(default_factory=list)

    @classmethod
    def from_api_record(cls, record: dict[str, Any]) -> "JobRaw":
        return cls(**record)

    # --- flattening helpers, used by fetch.py to build CSV rows ---

    @property
    def company_name(self) -> str | None:
        if isinstance(self.company, CompanyRef):
            return self.company.name
        if isinstance(self.company, str):
            return self.company
        return None

    @property
    def company_id(self) -> int | None:
        return self.company.id if isinstance(self.company, CompanyRef) else None

    @property
    def company_is_agency(self) -> bool | None:
        return self.company.is_agency if isinstance(self.company, CompanyRef) else None

    @property
    def topic_names(self) -> str:
        return "|".join(t.name for t in self.topics if t.name)

    @property
    def topic_ids(self) -> str:
        return "|".join(str(t.id) for t in self.topics if t.id is not None)

    @property
    def tag_names(self) -> str:
        return "|".join(t.name for t in self.tags if t.name)

    @property
    def region_names(self) -> str:
        return "|".join(r.name for r in self.regions if r.name)

    @property
    def country_names(self) -> str:
        return "|".join(c.name for c in self.countries if c.name)


class SalaryRaw(BaseModel):
    """Loose validation for records from /insight/salary/.
    Field names are placeholders pending confirmation 
    """
    model_config = ConfigDict(extra="allow")

    id: Any = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api_record(cls, record: dict[str, Any]) -> "SalaryRaw":
        return cls(id=record.get("id"), raw=record, **{k: v for k, v in record.items() if k != "id"})
