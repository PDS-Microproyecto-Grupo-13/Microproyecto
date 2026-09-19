from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from ml_pipeline.common.io import read_json
from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.pyfunc import OUTPUT_COLUMNS, create_input_example
from ml_pipeline.modeling.qualify import qualify
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.mlflow_tracker import track
from ml_pipeline.tracking.registry import register_candidate
import sys

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from test_pipeline import synthetic_foorilla_env  # noqa: E402


def test_track_and_register_candidate_contract(
    synthetic_foorilla_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = synthetic_foorilla_env
    # 1. Setup isolated temporary MLflow backend and artifact store
    db_path = tmp_path / "mlflow.db"
    artifacts_dir = tmp_path / "mlartifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"
    experiment_name = "test-salary-integration"
    model_name = "salary_predict_model-integration"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    monkeypatch.setenv("MLFLOW_MODEL_NAME", model_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(experiment_name, artifact_location=artifacts_dir.as_uri())

    settings = Settings.load(tmp_path)

    # 2. Run DVC pipeline stages through evaluate to produce an eligible model
    for stage in (collect, validate, preprocess, qualify, train, evaluate):
        stage(settings)

    # 3. Verify candidate is eligible before tracking
    candidate_report = tmp_path / "artifacts/reports/candidate.json"
    assert candidate_report.is_file()
    candidate_data = read_json(candidate_report)
    assert candidate_data["eligible"] is True

    # 4. Execute track
    run_id = track(settings)
    assert isinstance(run_id, str) and len(run_id) > 0

    # 5. Verify tracking.json
    tracking_file = tmp_path / "artifacts/reports/tracking.json"
    assert tracking_file.is_file()
    tracking_data = read_json(tracking_file)
    assert tracking_data["run_id"] == run_id
    assert tracking_data["algorithm"] == "lightgbm"
    assert tracking_data["eligible"] is True
    assert "model_uri" in tracking_data

    # 6. Verify MLflow Run in isolated backend
    run = client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.data.tags.get("algorithm") == "lightgbm"
    assert run.data.params.get("model.algorithm") == "lightgbm"

    # 7. Verify PyFunc model artifact exists and can be loaded
    loaded_model = mlflow.pyfunc.load_model(tracking_data["model_uri"])
    sample_features = create_input_example()
    source_predictions = loaded_model.predict(sample_features)
    assert len(source_predictions) == len(sample_features)
    for col in OUTPUT_COLUMNS:
        assert col in source_predictions.columns

    # 8. Execute register_candidate
    version_str = register_candidate(settings)
    assert version_str == "1"

    # 9. Verify registered model exists in isolated registry
    registered_model = client.get_registered_model(model_name)
    assert registered_model.name == model_name

    # 10. Verify model version associated with run_id and correct tags
    model_version = client.get_model_version(name=model_name, version=version_str)
    assert str(model_version.version) == version_str
    assert model_version.run_id == run_id
    assert model_version.tags.get("candidate") == "true"
    assert model_version.tags.get("eligible") == "true"
    assert model_version.tags.get("algorithm") == "lightgbm"
    assert getattr(model_version, "aliases", []) == []

    # 11. Verify registration.json generated
    registration_file = tmp_path / "artifacts/reports/registration.json"
    assert registration_file.is_file()
    registration_data = read_json(registration_file)
    assert registration_data["model_name"] == model_name
    assert registration_data["version"] == version_str
    assert registration_data["run_id"] == run_id
    assert registration_data["exact_registry_uri"] == f"models:/{model_name}/{version_str}"

    # 12. Verify registered model version loads and achieves parity
    registry_uri = f"models:/{model_name}/{version_str}"
    loaded_version_model = mlflow.pyfunc.load_model(registry_uri)
    reg_predictions = loaded_version_model.predict(sample_features)
    for col in OUTPUT_COLUMNS:
        assert np.allclose(source_predictions[col], reg_predictions[col], rtol=1e-5, atol=1e-5)

    # 13. Verify deduplication: re-running register_candidate returns same version and does not create version 2
    v2 = register_candidate(settings)
    assert v2 == "1"
    versions = client.search_model_versions(f"name = '{model_name}'")
    assert len(versions) == 1

