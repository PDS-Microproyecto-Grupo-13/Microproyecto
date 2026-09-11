from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from ml_pipeline.common.io import read_json, write_json
from ml_pipeline.modeling.pyfunc import OUTPUT_COLUMNS, create_input_example
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import collect_lineage

LOGGER = logging.getLogger(__name__)

__all__ = ["RegistrationError", "register_candidate"]


class RegistrationError(RuntimeError):
    """Raised when model registration preconditions or validations fail."""
    pass


def register_candidate(settings: Settings) -> str:
    """Register the candidate PyFunc model from Phase 6 into MLflow Model Registry.

    Execution flow:
    1. Validates presence and eligibility in candidate.json and tracking.json.
    2. Validates that the MLflow Run and PyFunc model exist and can be loaded.
    3. Validates dataset_fingerprint consistency.
    4. Checks for existing candidate version to prevent duplicate creation.
    5. Registers model version and applies metadata tags (no aliases, no stages).
    6. Verifies exact numerical parity between Tracking model and Registry version.
    7. Persists artifacts/reports/registration.json.
    """
    candidate_path = settings.path("artifacts/reports/candidate.json")
    tracking_path = settings.path("artifacts/reports/tracking.json")

    # 1. Precondition checks: file presence
    if not candidate_path.is_file():
        raise RegistrationError(f"candidate.json not found at {candidate_path}. Run evaluate first.")
    if not tracking_path.is_file():
        raise RegistrationError(f"tracking.json not found at {tracking_path}. Run track first.")

    candidate = read_json(candidate_path)
    tracking = read_json(tracking_path)

    # 2. Precondition checks: eligibility
    if candidate.get("eligible") is not True:
        raise RegistrationError(
            f"Model is not an eligible candidate in candidate.json: {candidate.get('reasons', [])}"
        )
    if tracking.get("eligible") is not True:
        raise RegistrationError("Model is not marked as eligible in tracking.json")

    run_id = tracking.get("run_id")
    model_uri = tracking.get("model_uri")
    if not run_id:
        raise RegistrationError("tracking.json does not contain run_id")
    if not model_uri:
        raise RegistrationError("tracking.json does not contain model_uri")

    # 3. Precondition checks: fingerprint & lineage consistency
    candidate_fp = candidate.get("dataset_fingerprint")
    tracking_fp = tracking.get("dataset_fingerprint")
    if candidate_fp and tracking_fp and candidate_fp != tracking_fp:
        raise RegistrationError(
            f"Dataset fingerprint mismatch: candidate.json ({candidate_fp}) vs tracking.json ({tracking_fp})"
        )
    dataset_fingerprint = tracking_fp or candidate_fp or ""

    if settings.require_clean_git:
        lineage = collect_lineage(settings)
        if lineage.get("git_dirty") is True:
            raise RegistrationError("Git working directory is dirty and ML_REQUIRE_CLEAN_GIT is true")

    model_name = settings.mlflow_model_name
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    # 4. Precondition checks: verify MLflow Run exists
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise RegistrationError(f"MLflow Run '{run_id}' not found at {settings.mlflow_tracking_uri}: {exc}") from exc

    # 5. Precondition checks: verify PyFunc model resolves and loads from tracking URI
    try:
        source_pyfunc = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise RegistrationError(f"Source PyFunc model '{model_uri}' cannot be loaded: {exc}") from exc

    # 6. Prevent accidental duplicates: check if version already registered for this run / model_uri
    existing_version: str | None = None
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
        for v in versions:
            if v.run_id == run_id or v.source == model_uri:
                existing_version = str(v.version)
                LOGGER.info(
                    "register-candidate | existing version found | model=%s version=%s run_id=%s",
                    model_name,
                    existing_version,
                    run_id,
                )
                break
    except Exception:
        pass

    if existing_version is not None:
        version_str = existing_version
    else:
        LOGGER.info("register-candidate | registering model %s from %s", model_name, model_uri)
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        version_str = str(model_version.version)

    # 7. Apply required tags to the candidate Model Version (no aliases, no stages)
    primary_metric = tracking.get("primary_metric", "mae_promedio")
    primary_metric_value = tracking.get(
        "primary_metric_value",
        candidate.get("test_evaluation", {}).get("primary_metric_value"),
    )
    tags: dict[str, str] = {
        "candidate": "true",
        "eligible": "true",
        "algorithm": str(tracking.get("algorithm", "lightgbm")),
        "dataset_fingerprint": str(dataset_fingerprint),
        "run_id": str(run_id),
        "primary_metric": str(primary_metric),
        "primary_metric_value": str(primary_metric_value),
    }
    for tag_key, tag_val in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version_str,
            key=tag_key,
            value=tag_val,
        )

    # 8. Mandatory Verification: load EXACT version from Registry and assert parity
    exact_registry_uri = f"models:/{model_name}/{version_str}"
    LOGGER.info("register-candidate | verifying exact registry model | uri=%s", exact_registry_uri)
    try:
        registry_pyfunc = mlflow.pyfunc.load_model(exact_registry_uri)
    except Exception as exc:
        raise RegistrationError(f"Failed to load registered model from {exact_registry_uri}: {exc}") from exc

    input_example = create_input_example()
    source_preds = source_pyfunc.predict(input_example)
    registry_preds = registry_pyfunc.predict(input_example)

    for col in OUTPUT_COLUMNS:
        if not np.allclose(source_preds[col], registry_preds[col], rtol=1e-5, atol=1e-5):
            raise RegistrationError(
                f"Parity mismatch on column '{col}' between source tracking model and registry model version"
            )

    reg_arr = registry_preds[OUTPUT_COLUMNS].to_numpy()
    if not np.isfinite(reg_arr).all():
        raise RegistrationError("Registered model produced non-finite values (NaN/Inf)")
    if (reg_arr <= 0).any():
        raise RegistrationError("Registered model produced non-positive predictions")
    if not (registry_preds["salary_min_usd"] <= registry_preds["salary_max_usd"]).all():
        raise RegistrationError("Registered model violated min <= max ordering")
    if not np.allclose(
        registry_preds["salary_midpoint_usd"],
        (registry_preds["salary_min_usd"] + registry_preds["salary_max_usd"]) / 2.0,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RegistrationError("Registered model midpoint is not the exact average of min and max")

    train_limits = candidate.get("training", {}).get("train_limits")
    if train_limits:
        floor = float(train_limits["floor"])
        ceiling = float(train_limits["ceiling"])
        if (registry_preds["salary_min_usd"] < floor - 1e-3).any() or (
            registry_preds["salary_max_usd"] > ceiling + 1e-3
        ).any():
            raise RegistrationError("Registered model output violates operational train limits")

    # 9. Persist local registration report
    registration_result: dict[str, Any] = {
        "model_name": model_name,
        "version": version_str,
        "run_id": run_id,
        "source": model_uri,
        "model_uri": model_uri,
        "dataset_fingerprint": dataset_fingerprint,
        "eligible": True,
        "primary_metric": primary_metric,
        "primary_metric_value": primary_metric_value,
        "exact_registry_uri": exact_registry_uri,
    }

    registration_path = settings.path("artifacts/reports/registration.json")
    write_json(registration_path, registration_result)

    LOGGER.info(
        "register-candidate | result=success | model=%s version=%s run_id=%s exact_uri=%s",
        model_name,
        version_str,
        run_id,
        exact_registry_uri,
    )

    return version_str
