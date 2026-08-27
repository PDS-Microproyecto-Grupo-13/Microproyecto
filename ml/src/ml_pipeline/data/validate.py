from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer

from ml_pipeline.common.io import ensure_parent, write_json
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)


class ValidationError(ValueError):
    pass


def expected_columns() -> list[str]:
    return [*load_breast_cancer().feature_names.tolist(), "target"]


def validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    expected = expected_columns()
    if frame.empty:
        errors.append("dataset is empty")
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing:
        errors.append(f"missing columns: {missing}")
    if unexpected:
        errors.append(f"unexpected columns: {unexpected}")
    if list(frame.columns) != expected and not missing and not unexpected:
        errors.append("columns are not in the expected order")
    non_numeric = [name for name in frame.columns if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        errors.append(f"non-numeric columns: {non_numeric}")
    if "target" in frame and not set(frame["target"].dropna().unique()).issubset({0, 1}):
        errors.append("target must contain only 0 and 1")
    if frame.isna().any().any():
        errors.append("dataset contains missing values")
    return {
        "valid": not errors,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "errors": errors,
    }


def validate(settings: Settings) -> None:
    source = settings.path("data/raw/dataset.csv")
    output = settings.path("data/validated/dataset.csv")
    report_path = settings.path("artifacts/reports/validation.json")
    LOGGER.info("validate | input=%s | output=%s", source, output)
    if not source.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {source}")
    frame = pd.read_csv(source)
    report = validate_frame(frame)
    write_json(report_path, report)
    if not report["valid"]:
        LOGGER.error("validate | result=failed | errors=%s", report["errors"])
        raise ValidationError("Dataset validation failed: " + "; ".join(report["errors"]))
    ensure_parent(output)
    frame.to_csv(output, index=False)
    LOGGER.info("validate | result=success | report=%s", report_path)
