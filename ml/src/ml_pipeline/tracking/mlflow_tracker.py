from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd

from ml_pipeline.common.io import read_json, write_json
from ml_pipeline.modeling.factory import effective_model_params
from ml_pipeline.modeling.pyfunc import (
    SalaryPredictorModel,
    build_model_signature,
    create_input_example,
    verify_local_pyfunc_parity,
)
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import as_mlflow_tags, collect_lineage

LOGGER = logging.getLogger(__name__)

__all__ = [
    "effective_tracking_params",
    "filter_inactive_algorithm_params",
    "normalize_mlflow_param",
    "track",
]


def normalize_mlflow_param(value: Any) -> Any:
    """Normalize None values to 'null' string for MLflow parameters."""
    if value is None:
        return "null"
    return value


def flatten_params(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dictionary parameters."""
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_params(item, name))
        else:
            flattened[name] = normalize_mlflow_param(item)
    return flattened


def filter_inactive_algorithm_params(params: dict[str, Any]) -> dict[str, Any]:
    """Filter parameters of inactive algorithms, retaining only the active algorithm's config."""
    filtered = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}
    model_config = filtered.get("model")
    if isinstance(model_config, dict):
        algorithm = model_config.get("algorithm")
        if algorithm:
            effective = effective_model_params(model_config)
            algo_params = {k: v for k, v in effective.items() if k != "random_state"}
            new_model: dict[str, Any] = {"algorithm": algorithm}
            if "random_state" in effective:
                new_model["random_state"] = effective["random_state"]
            new_model[algorithm] = algo_params
            filtered["model"] = new_model
    return filtered


def effective_tracking_params(params: dict[str, Any]) -> dict[str, Any]:
    """Assemble all effective parameters to be logged into MLflow Tracking."""
    tracking_params: dict[str, Any] = {}

    for section_name, section_value in params.items():
        if section_name == "model":
            continue
        if isinstance(section_value, dict):
            for key, val in flatten_params(section_value, prefix=section_name).items():
                tracking_params[key] = normalize_mlflow_param(val)
        else:
            tracking_params[section_name] = normalize_mlflow_param(section_value)

    model_config = params.get("model")
    if isinstance(model_config, dict):
        algorithm = model_config.get("algorithm")
        tracking_params["model.algorithm"] = normalize_mlflow_param(algorithm)
        tracking_params["model.feature_count"] = 24
        tracking_params["model.targets"] = "y_min_usd, y_max_usd"
        if algorithm:
            effective = effective_model_params(model_config)
            common_keys = {"random_state"}
            for key, val in effective.items():
                if key in common_keys:
                    param_name = f"model.{key}"
                else:
                    param_name = f"model.{algorithm}.{key}"
                tracking_params[param_name] = normalize_mlflow_param(val)

    return tracking_params


def track(settings: Settings) -> str:
    """Register an MLflow Tracking Run for the salary prediction model.

    Execution flow:
    1. Validates presence of model.joblib and evaluated reports from Phase 5.
    2. Builds PyFunc wrapper, input example, and model signature (raw -> named output).
    3. Verifies local vs PyFunc parity on input example.
    4. Starts MLflow Run:
       - Logs effective parameters (algorithm, hyperparameters, targets, thresholds).
       - Logs final test/validation metrics from Phase 5 reports.
       - Logs Git lineage, DVC revision, and data fingerprints as tags.
       - Logs all Phase 5 report artifacts.
       - Logs the PyFunc custom model under the 'model' artifact path with code_paths.
    5. Verifies post-logging parity by reloading model from MLflow model_uri.
    6. Persists artifacts/reports/tracking.json for future Model Registry phases.
    """
    model_path = settings.path("artifacts/work/model/model.joblib")
    metrics_path = settings.path("artifacts/reports/metrics.json")
    qual_path = settings.path("artifacts/reports/qualification.json")
    training_rep_path = settings.path("artifacts/reports/training.json")
    candidate_path = settings.path("artifacts/reports/candidate.json")
    manifest_path = settings.path("artifacts/reports/experiment_manifest.json")

    if not model_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Run pipeline stages through evaluate before track (missing {model_path} or {metrics_path})"
        )

    LOGGER.info("track | loading model bundle and reports | model=%s", model_path)
    bundle: dict[str, Any] = joblib.load(model_path)
    metrics_data = read_json(metrics_path)
    candidate_data = read_json(candidate_path) if candidate_path.is_file() else {}
    qual_data = read_json(qual_path) if qual_path.is_file() else {}
    training_data = read_json(training_rep_path) if training_rep_path.is_file() else {}

    model_config = settings.section("model")
    algorithm = str(model_config.get("algorithm", bundle.get("algorithm", "lightgbm")))
    dataset_fingerprint = str(
        candidate_data.get("dataset_fingerprint", bundle.get("dataset_fingerprint", ""))
    )
    eligible = bool(candidate_data.get("eligible", True))

    # 1. Prepare PyFunc wrapper, input example, and signature
    input_example = create_input_example()
    pyfunc_model = SalaryPredictorModel(bundle=bundle)
    signature = build_model_signature(pyfunc_model, input_example)

    # 2. Pre-logging local vs PyFunc parity check
    verify_local_pyfunc_parity(bundle, input_example, pyfunc_model)

    # 3. Configure MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    lineage = collect_lineage(settings, include_lock=True)

    LOGGER.info(
        "track | uri=%s | experiment=%s | algorithm=%s",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
        algorithm,
    )

    code_path = settings.root / "src" / "ml_pipeline"
    if not code_path.is_dir():
        import ml_pipeline
        code_path = Path(ml_pipeline.__file__).resolve().parent
    pip_requirements = [
        "mlflow==3.15.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.4.0",
        "lightgbm>=4.0.0",
        "joblib>=1.3.0",
    ]

    with mlflow.start_run(run_name=algorithm) as run:
        run_id = run.info.run_id

        # Log parameters
        tracking_params = effective_tracking_params(settings.params)
        mlflow.log_params(tracking_params)

        # Log metrics (consuming already produced Phase 5 results)
        raw_metrics: dict[str, Any] = {
            # Test metrics
            "test_mae_min": metrics_data.get("mae_min"),
            "test_mae_max": metrics_data.get("mae_max"),
            "test_mae_promedio": metrics_data.get("mae_promedio"),
            "test_rmse_min": metrics_data.get("rmse_min"),
            "test_rmse_max": metrics_data.get("rmse_max"),
            "test_mape_min": metrics_data.get("mape_min"),
            "test_mape_max": metrics_data.get("mape_max"),
            "test_r2_min": metrics_data.get("r2_min"),
            "test_r2_max": metrics_data.get("r2_max"),
            "test_mae_amplitud": metrics_data.get("mae_amplitud"),
            "test_cobertura_intervalo": metrics_data.get("cobertura_intervalo"),
            "test_incoherencia_raw": metrics_data.get("incoherencia_raw"),
            "test_prediccion_no_positiva": metrics_data.get("prediccion_no_positiva"),
            "uncertainty_margin": metrics_data.get("uncertainty_margin"),
            "uncertainty_nominal_coverage": metrics_data.get("uncertainty_nominal_coverage"),
            "uncertainty_test_coverage": metrics_data.get("uncertainty_test_coverage"),
            # Qualification & validation metrics
            "val_mae_promedio": qual_data.get("validation_mae"),
            "baseline_mae": qual_data.get("baseline_mae"),
            "improvement_vs_baseline": qual_data.get("improvement_vs_baseline"),
            "backtest_mae": qual_data.get("backtest_mae"),
            "temporal_gap": qual_data.get("temporal_gap"),
            # Dataset row counts
            "test_rows": candidate_data.get("test_evaluation", {}).get("test_rows"),
            "train_rows": candidate_data.get("training", {}).get("train_rows"),
            "validation_rows": candidate_data.get("training", {}).get("validation_rows"),
            "final_training_rows": candidate_data.get("training", {}).get("final_training_rows"),
        }
        numeric_metrics = {
            k: float(v)
            for k, v in raw_metrics.items()
            if v is not None and isinstance(v, (int, float))
        }
        mlflow.log_metrics(numeric_metrics)

        # Log lineage and metadata tags
        tags = {
            **as_mlflow_tags(lineage),
            "algorithm": algorithm,
            "candidate_eligible": str(eligible),
            "primary_metric": str(settings.params.get("evaluation", {}).get("primary_metric", "mae_promedio")),
            "dataset_fingerprint": dataset_fingerprint,
            "train_rows": str(training_data.get("train_rows", "")),
            "validation_rows": str(training_data.get("validation_rows", "")),
            "final_training_rows": str(training_data.get("final_training_rows", "")),
            "test_rows": str(candidate_data.get("test_evaluation", {}).get("test_rows", "")),
        }
        mlflow.set_tags(tags)

        # Log reports artifacts
        reports_dir = settings.path("artifacts/reports")
        report_files = [
            "qualification.json",
            "uncertainty_calibration.json",
            "training.json",
            "metrics.json",
            "candidate.json",
            "experiment_manifest.json",
            "audit_segments.json",
            "audit_novelty.json",
            "audit_sensitivity.json",
            "feature_importance.json",
        ]
        for name in report_files:
            file_path = reports_dir / name
            if file_path.is_file():
                mlflow.log_artifact(str(file_path), artifact_path="reports")

        # Log PyFunc model under logical artifact 'model'
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=SalaryPredictorModel(),
            artifacts={"model_bundle": str(model_path)},
            code_paths=[str(code_path)],
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
        )

    # 4. Post-logging verification: reload model from MLflow storage and test parity
    LOGGER.info("track | reloading model from %s to verify parity", model_info.model_uri)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    verify_local_pyfunc_parity(bundle, input_example, loaded_pyfunc)

    # 5. Persist tracking.json report for Phase 7
    tracking_result: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": run.info.experiment_id,
        "experiment_name": settings.mlflow_experiment_name,
        "model_uri": model_info.model_uri,
        "model_id": getattr(model_info, "model_id", None),
        "artifact_location": run.info.artifact_uri,
        "algorithm": algorithm,
        "dataset_fingerprint": dataset_fingerprint,
        "eligible": eligible,
        "primary_metric": "mae_promedio",
        "primary_metric_value": metrics_data.get("mae_promedio"),
    }

    tracking_path = settings.path("artifacts/reports/tracking.json")
    write_json(tracking_path, tracking_result)

    LOGGER.info(
        "track | result=success | algorithm=%s | run_id=%s | model_uri=%s | eligible=%s",
        algorithm,
        run_id,
        model_info.model_uri,
        eligible,
    )

    return run_id
