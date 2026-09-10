from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from ml_pipeline.common.io import ensure_parent, read_json, write_json
from ml_pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.settings import Settings

# Suppress sklearn 1.9 TargetEncoder deprecation warning for shuffle & random_state
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.preprocessing._target_encoder")

LOGGER = logging.getLogger(__name__)

TARGET_COLUMNS: Final[list[str]] = ["y_min_usd", "y_max_usd"]


def build_preprocessor(random_state: int = 42) -> ColumnTransformer:
    """Construct independent Scikit-Learn preprocessor for salary target modeling."""
    return ColumnTransformer(
        [
            (
                "categoricas",
                Pipeline([
                    ("imputar", SimpleImputer(strategy="most_frequent")),
                    (
                        "codificar",
                        TargetEncoder(
                            target_type="continuous",
                            smooth="auto",
                            cv=5,
                            shuffle=True,
                            random_state=random_state,
                        ),
                    ),
                ]),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numericas",
                SimpleImputer(strategy="median"),
                NUMERIC_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_lgbm_estimator(params: dict[str, Any], random_state: int = 42) -> LGBMRegressor:
    """Build LightGBM regressor with canonical hyperparameter mapping."""
    effective_params = dict(params)
    effective_params.setdefault("random_state", random_state)
    effective_params.setdefault("n_jobs", -1)
    return LGBMRegressor(**effective_params)


def build_target_pipeline(params: dict[str, Any], random_state: int = 42) -> Pipeline:
    """Build encapsulated pipeline coupling preprocessor and LightGBM regressor."""
    return Pipeline([
        ("preprocesamiento", build_preprocessor(random_state=random_state)),
        ("modelo", build_lgbm_estimator(params, random_state=random_state)),
    ])


def postprocess(raw_prediction: np.ndarray, limits: dict[str, float]) -> np.ndarray:
    """Postprocess salary predictions: clip within train limits and enforce min <= max."""
    clipped = np.clip(np.asarray(raw_prediction, dtype=float), limits["floor"], limits["ceiling"])
    return np.sort(clipped, axis=1)


def compute_metrics(
    y_true: np.ndarray,
    raw_prediction: np.ndarray,
    limits: dict[str, float],
) -> tuple[dict[str, float], np.ndarray]:
    """Compute standard regression metrics matching historical notebook logic."""
    prediction = postprocess(raw_prediction, limits)
    result: dict[str, float] = {}
    for j, label in enumerate(["min", "max"]):
        result[f"mae_{label}"] = float(mean_absolute_error(y_true[:, j], prediction[:, j]))
        result[f"rmse_{label}"] = float(mean_squared_error(y_true[:, j], prediction[:, j]) ** 0.5)
        result[f"mape_{label}"] = float(mean_absolute_percentage_error(y_true[:, j], prediction[:, j]))
        result[f"r2_{label}"] = float(r2_score(y_true[:, j], prediction[:, j]))
    result["mae_promedio"] = float((result["mae_min"] + result["mae_max"]) / 2.0)
    result["mae_amplitud"] = float(mean_absolute_error(y_true[:, 1] - y_true[:, 0], prediction[:, 1] - prediction[:, 0]))
    result["cobertura_intervalo"] = float(np.mean((y_true[:, 0] >= prediction[:, 0]) & (y_true[:, 1] <= prediction[:, 1])))
    result["incoherencia_raw"] = float(np.mean(raw_prediction[:, 0] > raw_prediction[:, 1]))
    result["prediccion_no_positiva"] = float(np.mean(prediction <= 0))
    return result, prediction


def compute_baseline_metrics(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    limits: dict[str, float],
) -> tuple[dict[str, float], np.ndarray]:
    """Compute DummyRegressor (median of log1p targets) baseline on validation set."""
    raw: list[np.ndarray] = []
    for j in range(2):
        dummy = DummyRegressor(strategy="median").fit(
            np.zeros((len(y_train), 1)),
            np.log1p(y_train[:, j]),
        )
        pred_log = dummy.predict(np.zeros((len(y_eval), 1)))
        raw.append(np.expm1(pred_log))
    raw_pred = np.column_stack(raw)
    return compute_metrics(y_eval, raw_pred, limits)


def qualify(settings: Settings) -> dict[str, Any]:
    """Execute the qualification stage for salary prediction LightGBM models."""
    train_path = settings.path("data/processed/train.parquet")
    val_path = settings.path("data/processed/validation.parquet")
    limits_path = settings.path("artifacts/reports/train_limits.json")
    qual_report_path = settings.path("artifacts/reports/qualification.json")
    calib_report_path = settings.path("artifacts/reports/uncertainty_calibration.json")
    preprocess_report_path = settings.path("artifacts/reports/preprocess.json")

    LOGGER.info("qualify | loading datasets | train=%s val=%s", train_path, val_path)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    limits_data = read_json(limits_path)
    limits: dict[str, float] = {
        "floor": float(limits_data["floor"]),
        "ceiling": float(limits_data["ceiling"]),
    }

    # Extract model and qualification configuration
    model_section = settings.section("model")
    algorithm = str(model_section.get("algorithm", "lightgbm"))
    random_state = int(model_section.get("random_state", 42))

    lgb_params: dict[str, Any] = {}
    if algorithm == "lightgbm":
        lgb_params = dict(model_section.get("lightgbm", {}))
    elif "lightgbm" in model_section:
        lgb_params = dict(model_section.get("lightgbm", {}))
    else:
        raise ValueError(f"Configuration section 'model.lightgbm' is required for qualification")

    qual_section = settings.params.get("qualification", {})
    min_improvement = float(qual_section.get("min_improvement_vs_baseline", 0.10))
    max_temporal_gap = float(qual_section.get("max_temporal_gap", 0.25))
    backtest_train_ratio = float(qual_section.get("backtest_train_ratio", 0.80))
    nominal_coverage = float(qual_section.get("uncertainty_quantile", 0.80))

    # Features and targets
    X_train = train_df[FEATURE_COLUMNS]
    X_val = val_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMNS].to_numpy()
    y_val = val_df[TARGET_COLUMNS].to_numpy()

    # 1. Baseline evaluation on validation
    LOGGER.info("qualify | evaluating baseline model")
    base_scores, _ = compute_baseline_metrics(y_train, y_val, limits)
    base_mae = base_scores["mae_promedio"]

    # 2. Train dual LightGBM pipelines on Train and evaluate on Validation
    LOGGER.info("qualify | training dual LightGBM models on train partition")
    pipe_min = build_target_pipeline(lgb_params, random_state=random_state)
    pipe_min.fit(X_train, np.log1p(train_df["y_min_usd"]))
    raw_pred_val_min = np.expm1(pipe_min.predict(X_val))

    pipe_max = build_target_pipeline(lgb_params, random_state=random_state)
    pipe_max.fit(X_train, np.log1p(train_df["y_max_usd"]))
    raw_pred_val_max = np.expm1(pipe_max.predict(X_val))

    raw_val = np.column_stack([raw_pred_val_min, raw_pred_val_max])
    val_scores, val_pred = compute_metrics(y_val, raw_val, limits)
    val_mae = val_scores["mae_promedio"]

    improvement_vs_baseline = float((base_mae - val_mae) / base_mae)
    improvement_criterion_met = bool(improvement_vs_baseline >= min_improvement)

    # 3. Retrotest temporal (Backtest) strictly inside Train partition
    LOGGER.info("qualify | running temporal retrotest on train partition")
    cut = int(backtest_train_ratio * len(train_df))
    X_back_train = X_train.iloc[:cut]
    y_back_train = train_df.iloc[:cut]
    X_back_eval = X_train.iloc[cut:]
    y_back_eval = train_df.iloc[cut:]

    back_pipe_min = build_target_pipeline(lgb_params, random_state=random_state)
    back_pipe_min.fit(X_back_train, np.log1p(y_back_train["y_min_usd"]))
    raw_pred_back_min = np.expm1(back_pipe_min.predict(X_back_eval))

    back_pipe_max = build_target_pipeline(lgb_params, random_state=random_state)
    back_pipe_max.fit(X_back_train, np.log1p(y_back_train["y_max_usd"]))
    raw_pred_back_max = np.expm1(back_pipe_max.predict(X_back_eval))

    raw_back = np.column_stack([raw_pred_back_min, raw_pred_back_max])
    back_scores, _ = compute_metrics(y_back_eval[TARGET_COLUMNS].to_numpy(), raw_back, limits)
    back_mae = back_scores["mae_promedio"]

    temporal_gap = float((val_mae - back_mae) / back_mae)
    temporal_stability_criterion_met = bool(abs(temporal_gap) <= max_temporal_gap)

    # 4. Uncertainty calibration on validation set
    LOGGER.info("qualify | calibrating uncertainty margin on validation set")
    joint_error = np.max(np.abs(y_val - val_pred), axis=1)
    uncertainty_margin = float(np.quantile(joint_error, nominal_coverage, method="higher"))

    # 5. Qualification decision
    eligible = bool(improvement_criterion_met and temporal_stability_criterion_met)
    reasons: list[str] = []
    if not improvement_criterion_met:
        reasons.append(
            f"improvement_vs_baseline={improvement_vs_baseline:.4f} is below threshold={min_improvement:.4f}"
        )
    if not temporal_stability_criterion_met:
        reasons.append(
            f"abs(temporal_gap)={abs(temporal_gap):.4f} exceeds threshold={max_temporal_gap:.4f}"
        )

    # Extract dataset fingerprint if available
    dataset_fingerprint = ""
    if preprocess_report_path.is_file():
        preprocess_meta = read_json(preprocess_report_path)
        dataset_fingerprint = str(preprocess_meta.get("dataset_fingerprint", ""))

    qualification_report: dict[str, Any] = {
        "algorithm": algorithm,
        "configuration": lgb_params,
        "baseline_mae": round(base_mae, 4),
        "validation_mae": round(val_mae, 4),
        "improvement_vs_baseline": round(improvement_vs_baseline, 6),
        "backtest_mae": round(back_mae, 4),
        "temporal_gap": round(temporal_gap, 6),
        "criteria": {
            "min_improvement_vs_baseline": min_improvement,
            "max_temporal_gap": max_temporal_gap,
            "improvement_criterion_met": improvement_criterion_met,
            "temporal_stability_criterion_met": temporal_stability_criterion_met,
        },
        "eligible": eligible,
        "reasons": reasons,
        "uncertainty_margin": round(uncertainty_margin, 4),
        "nominal_coverage": nominal_coverage,
        "detailed_validation_metrics": {k: round(v, 6) for k, v in val_scores.items()},
        "dataset_fingerprint": dataset_fingerprint,
    }

    calibration_report: dict[str, Any] = {
        "nominal_coverage": nominal_coverage,
        "quantile_method": "higher",
        "uncertainty_margin": round(uncertainty_margin, 4),
        "validation_rows": len(val_df),
        "source_split": "validation",
        "dataset_fingerprint": dataset_fingerprint,
    }

    ensure_parent(qual_report_path)
    write_json(qual_report_path, qualification_report)
    write_json(calib_report_path, calibration_report)

    LOGGER.info(
        "qualify | complete | eligible=%s improvement=%.2f%% gap=%.2f%% uncertainty_margin=%.2f",
        eligible,
        improvement_vs_baseline * 100,
        temporal_gap * 100,
        uncertainty_margin,
    )

    return qualification_report
