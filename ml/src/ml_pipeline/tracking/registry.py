from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml_pipeline.common.io import read_json, write_json
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import collect_lineage

LOGGER = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    pass


def register_candidate(settings: Settings) -> str:
    candidate_path = settings.path("artifacts/reports/candidate.json")
    tracking_path = settings.path("artifacts/reports/tracking.json")
    if not candidate_path.is_file() or not tracking_path.is_file():
        raise RegistrationError("candidate.json and tracking.json are required")
    candidate = read_json(candidate_path)
    if candidate.get("eligible") is not True:
        raise RegistrationError(f"Model is not an eligible candidate: {candidate.get('reasons', [])}")
    run_id = read_json(tracking_path).get("run_id")
    if not run_id:
        raise RegistrationError("tracking.json does not contain run_id")
    if settings.require_clean_git and collect_lineage(settings)["git_dirty"] is not False:
        raise RegistrationError("A clean Git working tree is required for registration")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="model")
    if not artifacts:
        raise RegistrationError(f"Run {run_id} has no model artifact")
    try:
        client.get_registered_model(settings.mlflow_model_name)
    except MlflowException:
        client.create_registered_model(settings.mlflow_model_name)
    version = client.create_model_version(
        name=settings.mlflow_model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        tags={"candidate": "true"},
    )
    result = {
        "model_name": settings.mlflow_model_name,
        "version": str(version.version),
        "run_id": run_id,
    }
    write_json(settings.path("artifacts/reports/registration.json"), result)
    LOGGER.info("register-candidate | result=success | model=%s version=%s", settings.mlflow_model_name, version.version)
    return str(version.version)
