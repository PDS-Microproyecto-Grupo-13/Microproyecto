from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ml_pipeline.common.io import ensure_parent, write_json
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)

ESSENTIAL_COLUMNS: list[str] = [
    "id",
    "published",
    "y_min_usd",
    "y_max_usd",
    "target_source",
]
VALID_TARGET_SOURCES: set[str] = {"reportado", "híbrido", "estimado"}


class ValidationError(ValueError):
    """Raised when data fails schema or domain integrity validation."""


def expected_columns() -> list[str]:
    """Return essential columns expected in the consolidated interim dataset."""
    return list(ESSENTIAL_COLUMNS)


def validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate tabular integrity, non-nullity, domain ranges, and source categories."""
    errors: list[str] = []

    if frame.empty:
        errors.append("dataset is empty")
        return {
            "valid": False,
            "rows": 0,
            "columns": int(frame.shape[1]),
            "target_source_counts": {},
            "target_statistics": {},
            "errors": errors,
        }

    missing_cols = [col for col in ESSENTIAL_COLUMNS if col not in frame.columns]
    if missing_cols:
        errors.append(f"missing essential columns: {missing_cols}")

    if "id" in frame.columns and frame["id"].isna().any():
        errors.append(f"null IDs found in {int(frame['id'].isna().sum())} rows")

    if "published" in frame.columns and frame["published"].isna().any():
        errors.append(f"null published timestamps found in {int(frame['published'].isna().sum())} rows")

    if "y_min_usd" in frame.columns:
        null_count = int(frame["y_min_usd"].isna().sum())
        if null_count > 0:
            errors.append(f"null y_min_usd found in {null_count} rows")
        else:
            non_positive = int((frame["y_min_usd"] <= 0).sum())
            if non_positive > 0:
                errors.append(f"non-positive y_min_usd found in {non_positive} rows")

    if "y_max_usd" in frame.columns:
        null_count = int(frame["y_max_usd"].isna().sum())
        if null_count > 0:
            errors.append(f"null y_max_usd found in {null_count} rows")
        else:
            non_positive = int((frame["y_max_usd"] <= 0).sum())
            if non_positive > 0:
                errors.append(f"non-positive y_max_usd found in {non_positive} rows")

    if "y_min_usd" in frame.columns and "y_max_usd" in frame.columns:
        valid_both = frame["y_min_usd"].notna() & frame["y_max_usd"].notna()
        inverted = int((frame.loc[valid_both, "y_min_usd"] > frame.loc[valid_both, "y_max_usd"]).sum())
        if inverted > 0:
            errors.append(f"inverted target salary ranges (y_min_usd > y_max_usd) found in {inverted} rows")

    if "target_source" in frame.columns:
        sources_present = set(frame["target_source"].dropna().unique())
        invalid_sources = sorted(sources_present - VALID_TARGET_SOURCES)
        if invalid_sources:
            errors.append(f"invalid target_source categories: {invalid_sources}")

    target_source_counts = (
        {source: int((frame["target_source"] == source).sum()) for source in sorted(VALID_TARGET_SOURCES)}
        if "target_source" in frame.columns
        else {}
    )

    target_statistics: dict[str, dict[str, float]] = {}
    for col in ["y_min_usd", "y_max_usd"]:
        if col in frame.columns and not frame[col].dropna().empty:
            series = frame[col].dropna().astype(float)
            target_statistics[col] = {
                "min": float(series.min()),
                "median": float(series.median()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
            }

    return {
        "valid": not errors,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "target_source_counts": target_source_counts,
        "target_statistics": target_statistics,
        "errors": errors,
    }


def validate(settings: Settings) -> None:
    source = settings.path("data/interim/foorilla_consolidated.parquet")
    output = settings.path("data/validated/dataset.parquet")
    report_path = settings.path("artifacts/reports/validation.json")

    LOGGER.info("validate | input=%s | output=%s", source, output)
    if not source.is_file():
        raise FileNotFoundError(f"Interim dataset not found: {source}")

    frame = pd.read_parquet(source, engine="pyarrow")
    report = validate_frame(frame)
    write_json(report_path, report)

    if not report["valid"]:
        LOGGER.error("validate | result=failed | errors=%s", report["errors"])
        raise ValidationError("Dataset validation failed: " + "; ".join(report["errors"]))

    ensure_parent(output)
    frame.to_parquet(output, index=False, engine="pyarrow")
    LOGGER.info("validate | result=success | rows=%d | report=%s", report["rows"], report_path)
