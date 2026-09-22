from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.analytics_schema import AnalyticsSummary
from ml_pipeline.common.io import ensure_parent, read_json
from ml_pipeline.features import SKILLS
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class AnalyticsInputs:
    data_manifest: dict[str, Any]
    validation: dict[str, Any]
    preprocess: dict[str, Any]
    metrics: dict[str, Any]
    experiment_manifest: dict[str, Any]
    splits: dict[str, pd.DataFrame]


def load_analytics_inputs(settings: Settings) -> AnalyticsInputs:
    reports = settings.path("artifacts/reports")
    report_paths = {
        "data_manifest": reports / "data_manifest.json",
        "validation": reports / "validation.json",
        "preprocess": reports / "preprocess.json",
        "metrics": reports / "metrics.json",
        "experiment_manifest": reports / "experiment_manifest.json",
    }
    missing = [str(path) for path in report_paths.values() if not path.is_file()]
    split_paths = {
        name: settings.path(f"data/processed/{name}.parquet") for name in SPLIT_NAMES
    }
    missing.extend(str(path) for path in split_paths.values() if not path.is_file())
    if missing:
        raise FileNotFoundError("Required analytics input(s) not found: " + ", ".join(missing))

    return AnalyticsInputs(
        data_manifest=read_json(report_paths["data_manifest"]),
        validation=read_json(report_paths["validation"]),
        preprocess=read_json(report_paths["preprocess"]),
        metrics=read_json(report_paths["metrics"]),
        experiment_manifest=read_json(report_paths["experiment_manifest"]),
        splits={name: pd.read_parquet(path) for name, path in split_paths.items()},
    )


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_fingerprints(inputs: AnalyticsInputs) -> str:
    lineage = inputs.experiment_manifest.get("lineage")
    if not isinstance(lineage, dict):
        raise TypeError("experiment_manifest.lineage must be an object")
    candidates = {
        "data_manifest": inputs.data_manifest.get("dataset_fingerprint"),
        "preprocess": inputs.preprocess.get("dataset_fingerprint"),
        "experiment_manifest": inputs.experiment_manifest.get("dataset_fingerprint"),
        "experiment_manifest.lineage": lineage.get("dataset_fingerprint"),
    }
    for source, value in candidates.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing dataset fingerprint in {source}")
    if len(set(candidates.values())) != 1:
        raise ValueError(f"Dataset fingerprints are inconsistent: {candidates}")
    return str(candidates["data_manifest"])


def _validate_and_combine_splits(inputs: AnalyticsInputs) -> pd.DataFrame:
    preprocess_splits = inputs.preprocess.get("split")
    if not isinstance(preprocess_splits, dict):
        raise TypeError("preprocess.split must be an object")

    required_columns = {
        "published",
        "y_min_usd",
        "y_max_usd",
        "experience_level",
        "work_mode",
        *(f"skill_{name}" for name in SKILLS),
    }
    frames: list[pd.DataFrame] = []
    for name in SPLIT_NAMES:
        frame = inputs.splits[name]
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{name} split is missing columns: {missing_columns}")
        split_report = preprocess_splits.get(name)
        if not isinstance(split_report, dict):
            raise TypeError(f"preprocess.split.{name} must be an object")
        expected_rows = _require_int(
            split_report.get("rows"), f"preprocess.split.{name}.rows", minimum=1
        )
        if len(frame) != expected_rows:
            raise ValueError(
                f"{name} split row count mismatch: parquet={len(frame)} report={expected_rows}"
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    expected_modelable = _require_int(
        inputs.preprocess.get("rows_modeling"), "preprocess.rows_modeling", minimum=1
    )
    if len(combined) != expected_modelable:
        raise ValueError(
            f"Modelable row count mismatch: splits={len(combined)} report={expected_modelable}"
        )

    published = pd.to_datetime(combined["published"], errors="coerce", utc=True)
    if published.isna().any():
        raise ValueError("Modelable population contains invalid published timestamps")
    combined = combined.copy()
    combined["published"] = published

    for target in ("y_min_usd", "y_max_usd"):
        values = pd.to_numeric(combined[target], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Modelable population contains invalid {target}")
        combined[target] = values.astype(float)
    if (combined["y_min_usd"] < 0).any() or (combined["y_max_usd"] < 0).any():
        raise ValueError("Modelable salaries must be non-negative")
    if (combined["y_min_usd"] > combined["y_max_usd"]).any():
        raise ValueError("Modelable salary ranges must satisfy y_min_usd <= y_max_usd")

    for skill in SKILLS:
        column = f"skill_{skill}"
        values = pd.to_numeric(combined[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{column} must contain only binary 0/1 values")
        combined[column] = values.astype(int)
    return combined


def validate_analytics_inputs(inputs: AnalyticsInputs) -> tuple[pd.DataFrame, str]:
    if inputs.validation.get("valid") is not True:
        raise ValueError("validation.json must report valid=true")
    _require_int(inputs.validation.get("rows"), "validation.rows", minimum=0)

    snapshots = inputs.data_manifest.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("data_manifest.snapshots must be a non-empty array")
    snapshot_count = _require_int(
        inputs.data_manifest.get("snapshot_count"), "data_manifest.snapshot_count", minimum=1
    )
    if snapshot_count != len(snapshots):
        raise ValueError("data_manifest.snapshot_count does not match snapshots length")
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict):
            raise TypeError(f"data_manifest.snapshots[{index}] must be an object")
        _require_int(snapshot.get("rows_raw"), f"snapshots[{index}].rows_raw", minimum=0)

    fingerprint = _validate_fingerprints(inputs)
    combined = _validate_and_combine_splits(inputs)
    return combined, fingerprint


def validate_analytics_config(settings: Settings) -> tuple[list[int], int]:
    config = settings.section("analytics")
    raw_bins = config.get("salary_bins_usd")
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ValueError("analytics.salary_bins_usd must be a non-empty list")
    bins = [
        _require_int(value, f"analytics.salary_bins_usd[{index}]", minimum=0)
        for index, value in enumerate(raw_bins)
    ]
    if bins[0] != 0 or bins != sorted(set(bins)):
        raise ValueError("analytics.salary_bins_usd must start at 0 and be strictly increasing")
    limit = _require_int(
        config.get("top_technologies_limit"),
        "analytics.top_technologies_limit",
        minimum=1,
    )
    return bins, limit


def _proportion(count: int, population: int) -> float:
    return round(count / population, 6)


def build_salary_bins(
    midpoint: pd.Series, boundaries: list[int], population: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, lower in enumerate(boundaries):
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else None
        mask = midpoint.ge(lower) if upper is None else midpoint.ge(lower) & midpoint.lt(upper)
        count = int(mask.sum())
        result.append(
            {
                "lower_bound_usd": lower,
                "upper_bound_usd": upper,
                "count": count,
                "proportion": _proportion(count, population),
            }
        )
    return result


def build_category_distribution(series: pd.Series, population: int) -> list[dict[str, Any]]:
    categories = series.fillna("desconocido").astype(str).str.strip().replace("", "desconocido")
    counts = ((str(category), int(count)) for category, count in categories.value_counts().items())
    ordered = sorted(counts, key=lambda item: (-item[1], item[0]))
    return [
        {
            "category": category,
            "count": count,
            "proportion": _proportion(count, population),
        }
        for category, count in ordered
    ]


def build_top_technologies(
    frame: pd.DataFrame, population: int, limit: int
) -> list[dict[str, Any]]:
    counts = [(skill, int(frame[f"skill_{skill}"].sum())) for skill in SKILLS]
    ordered = sorted(counts, key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {
            "technology": technology,
            "count": count,
            "proportion": _proportion(count, population),
        }
        for technology, count in ordered
    ]


def _utc_rfc3339(value: pd.Timestamp) -> str:
    timestamp = value.tz_convert("UTC") if value.tzinfo is not None else value.tz_localize("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def build_analytics_summary(
    inputs: AnalyticsInputs,
    frame: pd.DataFrame,
    dataset_fingerprint: str,
    salary_boundaries: list[int],
    technology_limit: int,
) -> AnalyticsSummary:
    population = len(frame)
    midpoint = (frame["y_min_usd"] + frame["y_max_usd"]) / 2.0
    snapshots = inputs.data_manifest["snapshots"]
    raw_rows = sum(int(snapshot["rows_raw"]) for snapshot in snapshots)
    test_rows = len(inputs.splits["test"])
    reported_test_rows = int(inputs.preprocess["split"]["test"]["rows"])
    if test_rows != reported_test_rows:
        raise ValueError("evaluation rows do not match preprocess test rows")

    lineage = inputs.experiment_manifest["lineage"]
    legacy_dvc_hash = lineage.get("dvc_revision")
    dvc_yaml_hash = lineage.get("dvc_yaml_hash", legacy_dvc_hash)
    manifest_fingerprint = inputs.experiment_manifest.get("dataset_fingerprint")
    if manifest_fingerprint != dataset_fingerprint:
        raise ValueError("Experiment manifest fingerprint contradicts data manifest")

    metrics = inputs.metrics
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": {
            "source": inputs.data_manifest.get("source"),
            "target_scope": inputs.preprocess.get("target_scope"),
            "counts": {
                "raw_snapshot_rows": raw_rows,
                "validated_rows": int(inputs.validation["rows"]),
                "modelable_rows": population,
            },
            "data_range": {
                "published_min": _utc_rfc3339(frame["published"].min()),
                "published_max": _utc_rfc3339(frame["published"].max()),
            },
            "salary_midpoint": {
                "currency": "USD",
                "period": "annual",
                "mean_usd": round(float(midpoint.mean()), 2),
                "median_usd": round(float(midpoint.median()), 2),
                "minimum_usd": round(float(midpoint.min()), 2),
                "maximum_usd": round(float(midpoint.max()), 2),
            },
            "salary_midpoint_distribution": build_salary_bins(
                midpoint, salary_boundaries, population
            ),
            "seniority_distribution": build_category_distribution(
                frame["experience_level"], population
            ),
            "work_mode_distribution": build_category_distribution(
                frame["work_mode"], population
            ),
            "top_technologies": build_top_technologies(
                frame, population, technology_limit
            ),
        },
        "model": {
            "algorithm": inputs.experiment_manifest.get("algorithm"),
            "evaluation_rows": test_rows,
            "mae_average_usd": round(
                _require_finite_number(metrics.get("mae_promedio"), "metrics.mae_promedio"),
                2,
            ),
            "r2_salary_min": round(
                _require_finite_number(metrics.get("r2_min"), "metrics.r2_min"), 6
            ),
            "r2_salary_max": round(
                _require_finite_number(metrics.get("r2_max"), "metrics.r2_max"), 6
            ),
            "predicted_range_coverage": round(
                _require_finite_number(
                    metrics.get("cobertura_intervalo"), "metrics.cobertura_intervalo"
                ),
                6,
            ),
            "uncertainty_margin_usd": round(
                _require_finite_number(
                    metrics.get("uncertainty_margin"), "metrics.uncertainty_margin"
                ),
                2,
            ),
            "uncertainty_nominal_coverage": round(
                _require_finite_number(
                    metrics.get("uncertainty_nominal_coverage"),
                    "metrics.uncertainty_nominal_coverage",
                ),
                6,
            ),
            "uncertainty_test_coverage": round(
                _require_finite_number(
                    metrics.get("uncertainty_test_coverage"),
                    "metrics.uncertainty_test_coverage",
                ),
                6,
            ),
        },
        "metadata": {
            "dataset_fingerprint": dataset_fingerprint,
            "snapshot_count": int(inputs.data_manifest["snapshot_count"]),
            "git_commit": lineage.get("git_commit"),
            "dvc_yaml_hash": dvc_yaml_hash,
            "params_hash": lineage.get("params_hash"),
        },
    }
    return AnalyticsSummary.model_validate(payload)


def serialize_summary(summary: AnalyticsSummary) -> bytes:
    payload = summary.model_dump(mode="json")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_summary_atomic(path: Path, summary: AnalyticsSummary) -> None:
    content = serialize_summary(summary)
    ensure_parent(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def analytics(settings: Settings) -> AnalyticsSummary:
    LOGGER.info("analytics | loading reproducible local inputs")
    inputs = load_analytics_inputs(settings)
    frame, fingerprint = validate_analytics_inputs(inputs)
    boundaries, technology_limit = validate_analytics_config(settings)
    summary = build_analytics_summary(
        inputs, frame, fingerprint, boundaries, technology_limit
    )
    output = settings.path("artifacts/reports/dashboard_summary.json")
    write_summary_atomic(output, summary)
    LOGGER.info(
        "analytics | result=success | rows=%d | technologies=%d | output=%s",
        summary.dataset.counts.modelable_rows,
        len(summary.dataset.top_technologies),
        output,
    )
    return summary
