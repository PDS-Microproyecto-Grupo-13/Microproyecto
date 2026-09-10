from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_pipeline.common.io import ensure_parent, read_json, write_json
from ml_pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.modeling.qualify import (
    build_lgbm_estimator,
    build_preprocessor,
    build_target_pipeline,
    postprocess,
)
from ml_pipeline.settings import Settings

# Suppress sklearn 1.9 TargetEncoder deprecation warning for shuffle & random_state
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.preprocessing._target_encoder")

LOGGER = logging.getLogger(__name__)

TARGET_COLUMNS: Final[list[str]] = ["y_min_usd", "y_max_usd"]

__all__ = [
    "TARGET_COLUMNS",
    "predict_range",
    "predict_with_uncertainty",
    "train",
]


def predict_range(
    features: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    """Compute deterministic salary range predictions [min, max] using the model bundle.

    Execution flow:
    1. Validates required feature columns contract.
    2. Runs inference through dual pipelines (pipeline_min and pipeline_max).
    3. Transforms log-space predictions back to USD scale (expm1).
    4. Clips predictions strictly within train limits [floor, ceiling].
    5. Enforces min <= max ordering invariant.
    """
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    missing = set(feature_columns) - set(features.columns)
    if missing:
        raise ValueError(f"Features DataFrame missing required columns: {sorted(missing)}")

    pipe_min: Pipeline = bundle["pipeline_min"]
    pipe_max: Pipeline = bundle["pipeline_max"]
    limits: dict[str, float] = bundle["train_limits"]

    X = features[feature_columns]
    pred_min_log = pipe_min.predict(X)
    pred_max_log = pipe_max.predict(X)

    raw_min = np.expm1(pred_min_log)
    raw_max = np.expm1(pred_max_log)
    raw = np.column_stack([raw_min, raw_max])

    prediction = postprocess(raw, limits)

    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values (NaN or Inf)")
    if (prediction <= 0).any():
        raise ValueError("Prediction contains non-positive values")

    return prediction


def predict_with_uncertainty(
    features: pd.DataFrame,
    bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute salary range predictions with uncertainty bounds [lower, upper]."""
    prediction = predict_range(features, bundle)
    limits: dict[str, float] = bundle["train_limits"]
    margin: float = float(bundle["uncertainty_margin"])

    lower = np.maximum(limits["floor"], prediction - margin)
    upper = np.minimum(limits["ceiling"], prediction + margin)

    return prediction, lower, upper


def train(settings: Settings) -> dict[str, Any]:
    """Execute final model training on concatenated train + validation partitions."""
    qual_path = settings.path("artifacts/reports/qualification.json")
    if not qual_path.is_file():
        raise RuntimeError(
            f"Training gate failed: qualification report not found at {qual_path}. "
            "Execute qualification stage first."
        )

    qual_data = read_json(qual_path)
    if not qual_data.get("eligible", False):
        reasons = qual_data.get("reasons", [])
        raise RuntimeError(
            f"Training gate failed: model configuration is not eligible for final training "
            f"(qualification.json -> eligible == {qual_data.get('eligible')}). "
            f"Reasons: {reasons}"
        )

    limits_path = settings.path("artifacts/reports/train_limits.json")
    calib_path = settings.path("artifacts/reports/uncertainty_calibration.json")

    if not limits_path.is_file():
        raise RuntimeError(f"Required artifact not found: {limits_path}")
    if not calib_path.is_file():
        raise RuntimeError(f"Required artifact not found: {calib_path}")

    limits_data = read_json(limits_path)
    calib_data = read_json(calib_path)

    limits: dict[str, Any] = {
        "floor": float(limits_data["floor"]),
        "ceiling": float(limits_data["ceiling"]),
        "source_split": str(limits_data.get("source_split", "train")),
    }
    uncertainty_margin = float(calib_data["uncertainty_margin"])
    nominal_coverage = float(calib_data.get("nominal_coverage", 0.80))
    dataset_fingerprint = str(qual_data.get("dataset_fingerprint", ""))

    train_path = settings.path("data/processed/train.parquet")
    val_path = settings.path("data/processed/validation.parquet")

    LOGGER.info("train | loading datasets | train=%s val=%s", train_path, val_path)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    train_rows = len(train_df)
    val_rows = len(val_df)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    final_training_rows = len(trainval_df)

    # Extract model configuration
    model_section = settings.section("model")
    algorithm = str(model_section.get("algorithm", "lightgbm"))
    random_state = int(model_section.get("random_state", 42))

    lgb_params: dict[str, Any] = {}
    if algorithm == "lightgbm":
        lgb_params = dict(model_section.get("lightgbm", {}))
    elif "lightgbm" in model_section:
        lgb_params = dict(model_section.get("lightgbm", {}))
    else:
        raise ValueError("Configuration section 'model.lightgbm' is required for training")

    X_trainval = trainval_df[FEATURE_COLUMNS]

    LOGGER.info("train | fitting pipeline_min on %d rows", final_training_rows)
    pipe_min = build_target_pipeline(lgb_params, random_state=random_state)
    pipe_min.fit(X_trainval, np.log1p(trainval_df["y_min_usd"]))

    LOGGER.info("train | fitting pipeline_max on %d rows", final_training_rows)
    pipe_max = build_target_pipeline(lgb_params, random_state=random_state)
    pipe_max.fit(X_trainval, np.log1p(trainval_df["y_max_usd"]))

    bundle: dict[str, Any] = {
        "version": "1.0.0",
        "algorithm": algorithm,
        "configuration": lgb_params,
        "pipeline_min": pipe_min,
        "pipeline_max": pipe_max,
        "train_limits": limits,
        "uncertainty_margin": uncertainty_margin,
        "nominal_coverage": nominal_coverage,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "targets": list(TARGET_COLUMNS),
        "random_state": random_state,
        "dataset_fingerprint": dataset_fingerprint,
    }

    training_report: dict[str, Any] = {
        "algorithm": algorithm,
        "configuration": lgb_params,
        "train_rows": train_rows,
        "validation_rows": val_rows,
        "final_training_rows": final_training_rows,
        "feature_count": len(FEATURE_COLUMNS),
        "qualification": {
            "eligible": True,
        },
        "train_limits": limits,
        "uncertainty_margin": round(uncertainty_margin, 4),
        "dataset_fingerprint": dataset_fingerprint,
        "targets": list(TARGET_COLUMNS),
        "random_state": random_state,
    }

    model_output = settings.path("artifacts/work/model/model.joblib")
    report_output = settings.path("artifacts/reports/training.json")

    ensure_parent(model_output)
    joblib.dump(bundle, model_output, compress=3)
    LOGGER.info("train | persisted model bundle | path=%s", model_output)

    ensure_parent(report_output)
    write_json(report_output, training_report)
    LOGGER.info("train | persisted training report | path=%s", report_output)

    return training_report
