from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class SalaryPredictionRequest(BaseModel):
    """Stable public contract for salary-range inference."""

    title: str = Field(min_length=2, max_length=160)
    experience_level: str = Field(pattern="^(EN|MI|SE|EX)$")
    experience_years: float | None = Field(default=None, ge=0, le=50)
    country: str = Field(min_length=2, max_length=100)
    region: str | None = Field(default=None, max_length=100)
    regions: str | None = Field(default=None, max_length=100)
    work_mode: int | None = Field(default=None, ge=1, le=3)
    is_remote: bool = False
    company: str | None = Field(default=None, max_length=160)
    company_is_agency: bool = False
    technologies: list[str] = Field(default_factory=list, max_length=30)
    topics: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("technologies", "topics")
    @classmethod
    def clean_terms(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    def to_mlflow_record(self) -> dict[str, object]:
        raw_region = self.regions if self.regions is not None else self.region
        resolved_region = (raw_region or "desconocido").strip() or "desconocido"
        return {
            "title": self.title.strip(),
            "company": (self.company or "Sin información").strip() or "Sin información",
            "company_is_agency": self.company_is_agency,
            "countries": self.country.strip(),
            "regions": resolved_region,
            "experience_level": self.experience_level,
            "experience_years": self.experience_years,
            "has_remote": self.is_remote,
            "work_mode": self.work_mode,
            "tags": "|".join(self.technologies),
            "published": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        }


class SalaryRange(BaseModel):
    minimum_usd: float = Field(ge=0)
    maximum_usd: float = Field(ge=0)
    midpoint_usd: float = Field(ge=0)


class ModelDeployment(BaseModel):
    name: str
    alias: str
    version: str | None = None


class SalaryPredictionResponse(BaseModel):
    prediction: SalaryRange
    model: ModelDeployment
    warnings: list[str] = Field(default_factory=list)
