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

    tracking = read_json(tracking_path)

    run_id = tracking.get("run_id")
    model_id = tracking.get("model_id")
    model_uri = tracking.get("model_uri")

    if not run_id:
        raise RegistrationError("tracking.json does not contain run_id")
    if not model_id:
        raise RegistrationError("tracking.json does not contain model_id")
    if not model_uri:
        raise RegistrationError("tracking.json does not contain model_uri") 

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    version = mlflow.register_model(
        model_uri=model_uri,
        name=settings.mlflow_model_name,
    )
    client.set_model_version_tag(
        name=settings.mlflow_model_name,
        version=str(version.version),
        key="candidate",
        value="true",
    )

    result = {
        "model_name": settings.mlflow_model_name,
        "version": str(version.version),
        "run_id": run_id,
    }
    write_json(settings.path("artifacts/reports/registration.json"), result)
    LOGGER.info("register-candidate | result=success | model=%s version=%s", settings.mlflow_model_name, version.version)
    return str(version.version)
