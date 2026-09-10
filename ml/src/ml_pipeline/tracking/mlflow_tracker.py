from __future__ import annotations

import logging
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from ml_pipeline.common.io import read_json, write_json
from ml_pipeline.modeling.factory import effective_model_params
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import as_mlflow_tags, collect_lineage

LOGGER = logging.getLogger(__name__)


def normalize_mlflow_param(value: Any) -> Any:
    if value is None:
        return "null"
    return value


def flatten_params(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_params(item, name))
        else:
            flattened[name] = normalize_mlflow_param(item)
    return flattened


def filter_inactive_algorithm_params(params: dict[str, Any]) -> dict[str, Any]:
    filtered = {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}
    model_config = filtered.get("model")
    if isinstance(model_config, dict):
        algorithm = model_config.get("algorithm")
        if algorithm:
            effective = effective_model_params(model_config)
            algo_params = {k: v for k, v in effective.items() if k != "random_state"}
            new_model = {"algorithm": algorithm}
            if "random_state" in effective:
                new_model["random_state"] = effective["random_state"]
            new_model[algorithm] = algo_params
            filtered["model"] = new_model
    return filtered


def effective_tracking_params(params: dict[str, Any]) -> dict[str, Any]:
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
    model_path = settings.path("artifacts/work/model/model.joblib")
    metrics_path = settings.path("artifacts/reports/metrics.json")
    test_path = settings.path("data/processed/test.csv")
    if not model_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Run train and evaluate before track")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    model = joblib.load(model_path)
    test = pd.read_csv(test_path)
    example = test.drop(columns="target").head(5)
    signature = infer_signature(example, model.predict(example))
    metrics = read_json(metrics_path)
    lineage = collect_lineage(settings, include_lock=True)
    model_config = settings.section("model")
    algorithm = str(model_config.get("algorithm", "unknown"))
    LOGGER.info("track | uri=%s | experiment=%s | algorithm=%s", settings.mlflow_tracking_uri, settings.mlflow_experiment_name, algorithm)

    with mlflow.start_run(run_name=algorithm) as run:
        mlflow.log_params(effective_tracking_params(settings.params))
        mlflow.log_metrics(metrics)
        tags = {
            **as_mlflow_tags(lineage),
            "algorithm": algorithm,
        }
        mlflow.set_tags(tags)
        reports = settings.path("artifacts/reports")
        for name in ("validation.json", "metrics.json", "candidate.json", "experiment_manifest.json"):
            path = reports / name
            if path.is_file():
                mlflow.log_artifact(path, artifact_path="reports")
        # This is the only sklearn-flavor-specific boundary in tracking.
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=example,
            # MLflow 3 uses skops by default. This pipeline owns the fitted
            # sklearn object and explicitly trusts the dtype it serializes.
            skops_trusted_types=["numpy.dtype"],
        )
        run_id = run.info.run_id

    tracking_result = {
        "run_id": run_id,
        "algorithm": algorithm,
        "model_id": model_info.model_id,
        "model_uri": model_info.model_uri,
    }

    write_json(
        settings.path("artifacts/reports/tracking.json"),
        tracking_result,
    )

    LOGGER.info(
        "track | result=success | algorithm=%s | run_id=%s | model_id=%s | model_uri=%s",
        algorithm,
        run_id,
        model_info.model_id,
        model_info.model_uri,
    )

    return run_id
