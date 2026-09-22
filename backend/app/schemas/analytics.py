from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"
PROPORTION_DECIMALS = 6
CANONICAL_TECHNOLOGIES = frozenset(
    {
        "aws",
        "azure",
        "docker",
        "gcp",
        "kubernetes",
        "machine_learning",
        "power_bi",
        "python",
        "pytorch",
        "spark",
        "sql",
        "tableau",
        "tensorflow",
    }
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def validate_utc_rfc3339(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("timestamp must be a string")
    if "T" not in value or not value.endswith(("Z", "+00:00")):
        raise ValueError("timestamp must be RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be RFC3339 UTC") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise ValueError("timestamp must be RFC3339 UTC")
    return value


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class AnalyticsCounts(StrictContractModel):
    raw_snapshot_rows: int = Field(ge=0)
    validated_rows: int = Field(ge=0)
    modelable_rows: int = Field(gt=0)


class DataRange(StrictContractModel):
    published_min: str
    published_max: str

    @field_validator("published_min", "published_max")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return validate_utc_rfc3339(value)

    @model_validator(mode="after")
    def validate_order(self) -> DataRange:
        start = datetime.fromisoformat(self.published_min.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.published_max.replace("Z", "+00:00"))
        if start > end:
            raise ValueError("published_min must be <= published_max")
        return self


class SalaryMidpoint(StrictContractModel):
    currency: Literal["USD"]
    period: Literal["annual"]
    mean_usd: float = Field(ge=0)
    median_usd: float = Field(ge=0)
    minimum_usd: float = Field(ge=0)
    maximum_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> SalaryMidpoint:
        if not self.minimum_usd <= self.median_usd <= self.maximum_usd:
            raise ValueError("salary midpoint median must be within minimum and maximum")
        if not self.minimum_usd <= self.mean_usd <= self.maximum_usd:
            raise ValueError("salary midpoint mean must be within minimum and maximum")
        return self


class SalaryBin(StrictContractModel):
    lower_bound_usd: int = Field(ge=0)
    upper_bound_usd: int | None = None
    count: int = Field(ge=0)
    proportion: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> SalaryBin:
        if self.upper_bound_usd is not None and self.upper_bound_usd <= self.lower_bound_usd:
            raise ValueError("salary bin upper bound must be greater than lower bound")
        return self


class CategoryCount(StrictContractModel):
    category: NonEmptyString
    count: int = Field(ge=0)
    proportion: float = Field(ge=0, le=1)


class TechnologyCount(StrictContractModel):
    technology: NonEmptyString
    count: int = Field(ge=0)
    proportion: float = Field(ge=0, le=1)

    @field_validator("technology")
    @classmethod
    def validate_canonical_technology(cls, value: str) -> str:
        if value not in CANONICAL_TECHNOLOGIES:
            raise ValueError(f"technology must be a canonical skill key: {value}")
        return value


class DatasetAnalytics(StrictContractModel):
    source: NonEmptyString
    target_scope: NonEmptyString
    counts: AnalyticsCounts
    data_range: DataRange
    salary_midpoint: SalaryMidpoint
    salary_midpoint_distribution: list[SalaryBin] = Field(min_length=1)
    seniority_distribution: list[CategoryCount] = Field(min_length=1)
    work_mode_distribution: list[CategoryCount] = Field(min_length=1)
    top_technologies: list[TechnologyCount]


class ModelMetrics(StrictContractModel):
    algorithm: NonEmptyString
    evaluation_rows: int = Field(gt=0)
    mae_average_usd: float = Field(ge=0)
    r2_salary_min: float
    r2_salary_max: float
    predicted_range_coverage: float = Field(ge=0, le=1)
    uncertainty_margin_usd: float = Field(ge=0)
    uncertainty_nominal_coverage: float = Field(ge=0, le=1)
    uncertainty_test_coverage: float = Field(ge=0, le=1)


class AnalyticsMetadata(StrictContractModel):
    dataset_fingerprint: Sha256
    snapshot_count: int = Field(gt=0)
    git_commit: GitSha | None
    dvc_yaml_hash: Sha256 | None
    params_hash: Sha256 | None


def _expected_proportion(count: int, population: int) -> float:
    return round(count / population, PROPORTION_DECIMALS)


def _validate_distribution(
    name: str,
    values: list[CategoryCount],
    population: int,
) -> None:
    if sum(item.count for item in values) != population:
        raise ValueError(f"{name} counts must sum to modelable_rows")
    categories = [item.category for item in values]
    if len(categories) != len(set(categories)):
        raise ValueError(f"{name} categories must be unique")
    if values != sorted(values, key=lambda item: (-item.count, item.category)):
        raise ValueError(f"{name} must be ordered by count DESC and category ASC")
    for item in values:
        if abs(item.proportion - _expected_proportion(item.count, population)) > 1e-6:
            raise ValueError(f"{name} proportion does not match count/modelable_rows")
        if item.proportion != round(item.proportion, PROPORTION_DECIMALS):
            raise ValueError(f"{name} proportion must be rounded to 6 decimals")


class AnalyticsSummary(StrictContractModel):
    schema_version: Literal["1.0"]
    dataset: DatasetAnalytics
    model: ModelMetrics
    metadata: AnalyticsMetadata

    @model_validator(mode="after")
    def validate_cross_field_invariants(self) -> AnalyticsSummary:
        population = self.dataset.counts.modelable_rows
        bins = self.dataset.salary_midpoint_distribution
        if sum(item.count for item in bins) != population:
            raise ValueError("salary bin counts must sum to modelable_rows")
        if bins[0].lower_bound_usd != 0:
            raise ValueError("first salary bin must start at zero")
        for index, item in enumerate(bins):
            is_last = index == len(bins) - 1
            if is_last != (item.upper_bound_usd is None):
                raise ValueError("only the last salary bin may have an open upper bound")
            if not is_last and item.upper_bound_usd != bins[index + 1].lower_bound_usd:
                raise ValueError("salary bins must be ordered and contiguous")
            expected = _expected_proportion(item.count, population)
            if abs(item.proportion - expected) > 1e-6:
                raise ValueError("salary bin proportion does not match count/modelable_rows")
            if item.proportion != round(item.proportion, PROPORTION_DECIMALS):
                raise ValueError("salary bin proportion must be rounded to 6 decimals")

        _validate_distribution(
            "seniority_distribution", self.dataset.seniority_distribution, population
        )
        _validate_distribution(
            "work_mode_distribution", self.dataset.work_mode_distribution, population
        )

        technologies = self.dataset.top_technologies
        keys = [item.technology for item in technologies]
        if len(keys) != len(set(keys)):
            raise ValueError("top technologies must not contain duplicates")
        if technologies != sorted(technologies, key=lambda item: (-item.count, item.technology)):
            raise ValueError("top technologies must be ordered by count DESC and technology ASC")
        for item in technologies:
            if item.count > population:
                raise ValueError("technology count cannot exceed modelable_rows")
            expected = _expected_proportion(item.count, population)
            if abs(item.proportion - expected) > 1e-6:
                raise ValueError("technology proportion does not match count/modelable_rows")
            if item.proportion != round(item.proportion, PROPORTION_DECIMALS):
                raise ValueError("technology proportion must be rounded to 6 decimals")
        return self


class AnalyticsPublishResponse(StrictContractModel):
    artifact_hash: Sha256
    schema_version: Literal["1.0"]
    status: Literal["created", "replaced", "unchanged"]
    stored_at: str

    @field_validator("stored_at")
    @classmethod
    def validate_stored_at(cls, value: str) -> str:
        return validate_utc_rfc3339(value)


class AnalyticsStorageEnvelope(StrictContractModel):
    artifact_hash: Sha256
    storage_version: Literal[1]
    stored_at: str
    summary: AnalyticsSummary

    @field_validator("stored_at")
    @classmethod
    def validate_stored_at(cls, value: str) -> str:
        return validate_utc_rfc3339(value)
