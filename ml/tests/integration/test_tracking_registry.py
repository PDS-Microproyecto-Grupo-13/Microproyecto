from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient

from ml_pipeline.common.io import read_json
from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.mlflow_tracker import track
from ml_pipeline.tracking.registry import register_candidate


def test_track_and_register_candidate_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Setup isolated temporary MLflow backend and artifact store
    db_path = tmp_path / "mlflow.db"
    artifacts_dir = tmp_path / "mlartifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"
    experiment_name = "test-experiment-integration"
    model_name = "test-model-integration"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    monkeypatch.setenv("MLFLOW_MODEL_NAME", model_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(experiment_name, artifact_location=artifacts_dir.as_uri())

    # 2. Setup project params with baseline logistic_regression
    (tmp_path / "params.yaml").write_text(
        """data:
  test_size: 0.2
  random_state: 42
model:
  algorithm: logistic_regression
  random_state: 42
  logistic_regression:
    C: 1.0
    max_iter: 1000
evaluation:
  primary_metric: f1
  minimum_score: 0.8
""",
        encoding="utf-8",
    )

    settings = Settings.load(tmp_path)

    # 3. Run pipeline stages to produce an evaluated eligible model
    for stage in (collect, validate, preprocess, train, evaluate):
        stage(settings)

    # 4. Verify candidate is eligible before tracking
    candidate_report = tmp_path / "artifacts/reports/candidate.json"
    assert candidate_report.is_file()
    candidate_data = read_json(candidate_report)
    assert candidate_data["eligible"] is True

    # 5. Execute track
    run_id = track(settings)
    assert isinstance(run_id, str) and len(run_id) > 0

    # 6. Verify tracking.json
    tracking_file = tmp_path / "artifacts/reports/tracking.json"
    assert tracking_file.is_file()
    tracking_data = read_json(tracking_file)
    assert tracking_data["run_id"] == run_id
    assert tracking_data["algorithm"] == "logistic_regression"
    assert "model_id" in tracking_data
    assert "model_uri" in tracking_data

    # 7. Verify MLflow Run in isolated backend
    run = client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.data.tags.get("algorithm") == "logistic_regression"
    assert run.data.params.get("model.algorithm") == "logistic_regression"
    assert run.data.params.get("model.logistic_regression.C") == "1.0"
    assert run.data.params.get("model.logistic_regression.max_iter") == "1000"
    assert not any(k.startswith("model.random_forest") for k in run.data.params)

    # 8. Verify model artifact exists and can be retrieved / loaded
    model_info = mlflow.models.get_model_info(tracking_data["model_uri"])
    assert model_info.model_id == tracking_data["model_id"]
    assert model_info.run_id == run_id
    assert "sklearn" in model_info.flavors

    loaded_model = mlflow.sklearn.load_model(tracking_data["model_uri"])
    assert hasattr(loaded_model, "predict")

    test_frame = pd.read_csv(tmp_path / "data/processed/test.csv")
    sample_features = test_frame.drop(columns="target").head(5)
    predictions = loaded_model.predict(sample_features)
    assert len(predictions) == len(sample_features)

    # 9. Execute register_candidate
    version_str = register_candidate(settings)
    assert isinstance(version_str, str) and len(version_str) > 0

    # 10. Verify registered model exists in isolated registry
    registered_model = client.get_registered_model(model_name)
    assert registered_model.name == model_name

    # 11. Verify model version associated with run_id
    model_version = client.get_model_version(name=model_name, version=version_str)
    assert str(model_version.version) == version_str
    assert model_version.run_id == run_id

    # 12. Verify candidate=true tag preserved on Model Version
    assert model_version.tags.get("candidate") == "true"

    # 13. Verify registration.json generated
    registration_file = tmp_path / "artifacts/reports/registration.json"
    assert registration_file.is_file()

    # 14. Verify registration.json schema and contents
    registration_data = read_json(registration_file)
    assert registration_data["model_name"] == model_name
    assert registration_data["version"] == version_str
    assert registration_data["run_id"] == run_id

    # 15. Verify run_id matches between tracking and registration
    assert registration_data["run_id"] == tracking_data["run_id"]

    # 16. Verify registered model version can be loaded for inference
    registry_uri = f"models:/{model_name}/{version_str}"
    loaded_version_model = mlflow.sklearn.load_model(registry_uri)
    reg_predictions = loaded_version_model.predict(sample_features)
    assert (predictions == reg_predictions).all()
