from __future__ import annotations

import logging
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from ml_pipeline.common.io import read_json, write_json
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import as_mlflow_tags, collect_lineage

LOGGER = logging.getLogger(__name__)


def flatten_params(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_params(item, name))
        else:
            flattened[name] = item
    return flattened


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
    LOGGER.info("track | uri=%s | experiment=%s", settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params(flatten_params(settings.params))
        mlflow.log_metrics(metrics)
        mlflow.set_tags(as_mlflow_tags(lineage))
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
        "model_id": model_info.model_id,
        "model_uri": model_info.model_uri,
    }

    write_json(
        settings.path("artifacts/reports/tracking.json"),
        tracking_result,
    )

    LOGGER.info(
        "track | result=success | run_id=%s | model_id=%s | model_uri=%s",
        run_id,
        model_info.model_id,
        model_info.model_uri,
    )

    return run_id
