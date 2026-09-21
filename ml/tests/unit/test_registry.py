from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml_pipeline.common.io import write_json
from ml_pipeline.features import FEATURE_COLUMNS
from ml_pipeline.modeling import pyfunc as pyfunc_module
from ml_pipeline.modeling.pyfunc import SalaryPredictorModel, OUTPUT_COLUMNS, create_input_example
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.registry import RegistrationError, register_candidate


def create_mock_salary_bundle() -> dict:
    X_dummy = pd.DataFrame(0.0, index=range(10), columns=FEATURE_COLUMNS)
    for col in ["title", "country", "region", "experience_level", "work_mode", "company"]:
        X_dummy[col] = "known_val"

    pipe_min = Pipeline([
        ("preprocesamiento", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])),
        ("modelo", DummyRegressor(strategy="constant", constant=np.log1p(45000.0))),
    ])
    pipe_min.fit(X_dummy, np.log1p(np.array([45000.0] * 10)))

    pipe_max = Pipeline([
        ("preprocesamiento", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])),
        ("modelo", DummyRegressor(strategy="constant", constant=np.log1p(75000.0))),
    ])
    pipe_max.fit(X_dummy, np.log1p(np.array([75000.0] * 10)))

    return {
        "pipeline_min": pipe_min,
        "pipeline_max": pipe_max,
        "train_limits": {"floor": 10935.57, "ceiling": 720000.0, "source_split": "train"},
        "uncertainty_margin": 50927.2876,
        "nominal_coverage": 0.80,
        "feature_columns": FEATURE_COLUMNS,
        "algorithm": "lightgbm",
        "dataset_fingerprint": "mock_fp_registry",
    }


@pytest.fixture
def mock_mlflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "mlflow.db"
    artifacts_dir = tmp_path / "mlartifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"
    experiment_name = "test-salary-registry"
    model_name = "salary_predict_model-test"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    monkeypatch.setenv("MLFLOW_MODEL_NAME", model_name)

    (tmp_path / "params.yaml").write_text(
        """data: {}
model:
  algorithm: lightgbm
evaluation:
  primary_metric: mae_promedio
""",
        encoding="utf-8",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = client.create_experiment(experiment_name, artifact_location=artifacts_dir.as_uri())

    bundle = create_mock_salary_bundle()
    model_dir = tmp_path / "artifacts" / "work" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(bundle, model_path)

    code_path = Path(pyfunc_module.__file__).resolve()

    input_example = create_input_example()
    preds = SalaryPredictorModel(bundle=bundle).predict(None, input_example)
    sig = infer_signature(input_example, preds)

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=exp_id) as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SalaryPredictorModel(),
            artifacts={"model_bundle": str(model_path)},
            code_paths=[str(code_path)],
            signature=sig,
            input_example=input_example,
        )

    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    candidate_data = {
        "algorithm": "lightgbm",
        "dataset_fingerprint": "mock_fp_registry",
        "eligible": True,
        "training": {
            "train_limits": bundle["train_limits"],
        },
        "test_evaluation": {
            "primary_metric": "mae_promedio",
            "primary_metric_value": 27081.22,
        },
    }
    write_json(reports_dir / "candidate.json", candidate_data)

    tracking_data = {
        "run_id": run.info.run_id,
        "experiment_id": exp_id,
        "experiment_name": experiment_name,
        "model_uri": model_info.model_uri,
        "model_id": getattr(model_info, "model_id", None),
        "artifact_location": run.info.artifact_uri,
        "algorithm": "lightgbm",
        "dataset_fingerprint": "mock_fp_registry",
        "eligible": True,
        "primary_metric": "mae_promedio",
        "primary_metric_value": 27081.22,
    }
    write_json(reports_dir / "tracking.json", tracking_data)

    settings = Settings.load(tmp_path)
    return {
        "settings": settings,
        "client": client,
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "model_name": model_name,
        "tracking_uri": tracking_uri,
        "reports_dir": reports_dir,
        "tmp_path": tmp_path,
    }


def test_registration_rejects_missing_candidate_report(mock_mlflow_env) -> None:
    (mock_mlflow_env["reports_dir"] / "candidate.json").unlink()
    with pytest.raises(RegistrationError, match="candidate.json not found"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_missing_tracking_report(mock_mlflow_env) -> None:
    (mock_mlflow_env["reports_dir"] / "tracking.json").unlink()
    with pytest.raises(RegistrationError, match="tracking.json not found"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_ineligible_candidate(mock_mlflow_env) -> None:
    write_json(
        mock_mlflow_env["reports_dir"] / "candidate.json",
        {"eligible": False, "reasons": ["insufficient improvement"]},
    )
    with pytest.raises(RegistrationError, match="not an eligible candidate"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_ineligible_tracking(mock_mlflow_env) -> None:
    tracking_path = mock_mlflow_env["reports_dir"] / "tracking.json"
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["eligible"] = False
    write_json(tracking_path, data)

    with pytest.raises(RegistrationError, match="not marked as eligible in tracking.json"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_missing_run_id_or_model_uri(mock_mlflow_env) -> None:
    tracking_path = mock_mlflow_env["reports_dir"] / "tracking.json"
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Missing run_id
    data_no_run = dict(data)
    data_no_run.pop("run_id")
    write_json(tracking_path, data_no_run)
    with pytest.raises(RegistrationError, match="does not contain run_id"):
        register_candidate(mock_mlflow_env["settings"])

    # Missing model_uri
    data_no_uri = dict(data)
    data_no_uri.pop("model_uri")
    write_json(tracking_path, data_no_uri)
    with pytest.raises(RegistrationError, match="does not contain model_uri"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_fingerprint_mismatch(mock_mlflow_env) -> None:
    tracking_path = mock_mlflow_env["reports_dir"] / "tracking.json"
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["dataset_fingerprint"] = "conflicting_fingerprint"
    write_json(tracking_path, data)

    with pytest.raises(RegistrationError, match="Dataset fingerprint mismatch"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_nonexistent_run(mock_mlflow_env) -> None:
    tracking_path = mock_mlflow_env["reports_dir"] / "tracking.json"
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["run_id"] = "nonexistent_run_id_9999"
    write_json(tracking_path, data)

    with pytest.raises(RegistrationError, match="MLflow Run 'nonexistent_run_id_9999' not found"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_rejects_unresolvable_model(mock_mlflow_env) -> None:
    tracking_path = mock_mlflow_env["reports_dir"] / "tracking.json"
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["model_uri"] = "models:/nonexistent-model/999"
    write_json(tracking_path, data)

    with pytest.raises(RegistrationError, match="Source PyFunc model '.*' cannot be loaded"):
        register_candidate(mock_mlflow_env["settings"])


def test_registration_success_end_to_end_and_tags_and_reports(mock_mlflow_env) -> None:
    settings = mock_mlflow_env["settings"]
    client = mock_mlflow_env["client"]
    model_name = mock_mlflow_env["model_name"]
    run_id = mock_mlflow_env["run_id"]

    version_str = register_candidate(settings)
    assert version_str == "1"

    # Verify registered model in MLflow Model Registry
    reg_model = client.get_registered_model(model_name)
    assert reg_model.name == model_name

    model_version = client.get_model_version(model_name, version_str)
    assert str(model_version.version) == "1"
    assert model_version.run_id == run_id

    # Verify tags applied
    tags = model_version.tags
    assert tags.get("candidate") == "true"
    assert tags.get("eligible") == "true"
    assert tags.get("algorithm") == "lightgbm"
    assert tags.get("dataset_fingerprint") == "mock_fp_registry"
    assert tags.get("run_id") == run_id
    assert tags.get("primary_metric") == "mae_promedio"
    assert tags.get("primary_metric_value") == "27081.22"

    # Verify NO champion alias or production stage
    assert getattr(model_version, "aliases", []) == []
    assert model_version.current_stage in ("None", "")

    # Verify exact registry URI loads
    exact_uri = f"models:/{model_name}/{version_str}"
    loaded = mlflow.pyfunc.load_model(exact_uri)
    preds = loaded.predict(create_input_example())
    for col in OUTPUT_COLUMNS:
        assert col in preds.columns

    # Verify registration.json persisted
    reg_json_path = settings.path("artifacts/reports/registration.json")
    assert reg_json_path.is_file()
    with open(reg_json_path, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    assert reg_data["model_name"] == model_name
    assert reg_data["version"] == "1"
    assert reg_data["run_id"] == run_id
    assert reg_data["exact_registry_uri"] == exact_uri
    assert reg_data["eligible"] is True
    assert reg_data["primary_metric"] == "mae_promedio"
    assert reg_data["primary_metric_value"] == 27081.22


def test_registration_deduplication_prevents_duplicate_versions(mock_mlflow_env) -> None:
    settings = mock_mlflow_env["settings"]
    client = mock_mlflow_env["client"]
    model_name = mock_mlflow_env["model_name"]

    v1 = register_candidate(settings)
    assert v1 == "1"

    # Second invocation with exact same run_id / model_uri
    v2 = register_candidate(settings)
    assert v2 == "1"

    versions = client.search_model_versions(f"name = '{model_name}'")
    assert len(versions) == 1
    assert str(versions[0].version) == "1"


def test_registration_parity_failure_raises(mock_mlflow_env) -> None:
    settings = mock_mlflow_env["settings"]

    class CorruptPyFunc:
        def predict(self, df):
            # return dataframe with negative values
            return pd.DataFrame({
                "salary_min_usd": [-1000.0] * len(df),
                "salary_max_usd": [-500.0] * len(df),
                "salary_midpoint_usd": [-750.0] * len(df),
                "salary_min_confidence_usd": [-2000.0] * len(df),
                "salary_max_confidence_usd": [1000.0] * len(df),
            })

    original_load = mlflow.pyfunc.load_model

    def mock_load(uri):
        if str(uri).startswith("models:"):
            return CorruptPyFunc()
        return original_load(uri)

    with patch("mlflow.pyfunc.load_model", side_effect=mock_load):
        with pytest.raises(RegistrationError, match="Parity mismatch|non-positive"):
            register_candidate(settings)
