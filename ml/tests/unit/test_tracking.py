from __future__ import annotations

from pathlib import Path

import pytest
import mlflow
from mlflow.tracking import MlflowClient

from ml_pipeline.common.io import read_json
from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.mlflow_tracker import (
    effective_tracking_params,
    filter_inactive_algorithm_params,
    track,
)


def test_effective_tracking_params_for_logistic_regression() -> None:
    params = {
        "data": {"test_size": 0.2, "random_state": 42},
        "model": {
            "algorithm": "logistic_regression",
            "random_state": 42,
            "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "random_forest": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
        },
        "evaluation": {"primary_metric": "f1", "minimum_score": 0.8},
    }

    filtered = filter_inactive_algorithm_params(params)
    assert "random_forest" not in filtered["model"]
    assert "logistic_regression" in filtered["model"]

    effective = effective_tracking_params(params)
    assert effective["model.algorithm"] == "logistic_regression"
    assert effective["model.random_state"] == 42
    assert effective["model.logistic_regression.C"] == 1.0
    assert effective["model.logistic_regression.max_iter"] == 1000
    assert not any(k.startswith("model.random_forest") for k in effective)
    assert effective["data.test_size"] == 0.2
    assert effective["data.random_state"] == 42
    assert effective["evaluation.primary_metric"] == "f1"
    assert effective["evaluation.minimum_score"] == 0.8


def test_effective_tracking_params_for_random_forest() -> None:
    params = {
        "data": {"test_size": 0.2, "random_state": 42},
        "model": {
            "algorithm": "random_forest",
            "random_state": 42,
            "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "random_forest": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
        },
        "evaluation": {"primary_metric": "f1", "minimum_score": 0.8},
    }

    filtered = filter_inactive_algorithm_params(params)
    assert "logistic_regression" not in filtered["model"]
    assert "random_forest" in filtered["model"]

    effective = effective_tracking_params(params)
    assert effective["model.algorithm"] == "random_forest"
    assert effective["model.random_state"] == 42
    assert effective["model.random_forest.n_estimators"] == 200
    assert effective["model.random_forest.max_depth"] == 10
    assert effective["model.random_forest.min_samples_leaf"] == 1
    assert not any(k.startswith("model.logistic_regression") for k in effective)
    assert effective["data.test_size"] == 0.2
    assert effective["data.random_state"] == 42
    assert effective["evaluation.primary_metric"] == "f1"
    assert effective["evaluation.minimum_score"] == 0.8


@pytest.mark.parametrize("algorithm", ["logistic_regression", "random_forest"])
def test_track_creates_mlflow_run_with_filtered_params_and_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, algorithm: str
) -> None:
    db_path = tmp_path / "mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", f"test-experiment-{algorithm}")

    (tmp_path / "params.yaml").write_text(
        f"""data:
  test_size: 0.2
  random_state: 42
model:
  algorithm: {algorithm}
  random_state: 42
  logistic_regression:
    C: 1.0
    max_iter: 1000
  random_forest:
    n_estimators: 200
    max_depth: 10
    min_samples_leaf: 1
evaluation:
  primary_metric: f1
  minimum_score: 0.8
""",
        encoding="utf-8",
    )
    settings = Settings.load(tmp_path)
    for stage in (collect, validate, preprocess, train, evaluate):
        stage(settings)

    run_id = track(settings)
    assert isinstance(run_id, str) and len(run_id) > 0

    tracking_data = read_json(tmp_path / "artifacts/reports/tracking.json")
    assert tracking_data["run_id"] == run_id
    assert tracking_data["algorithm"] == algorithm
    assert "model_id" in tracking_data
    assert "model_uri" in tracking_data

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    run = client.get_run(run_id)

    assert run.info.run_name == algorithm
    assert run.data.tags["algorithm"] == algorithm
    assert run.data.params["model.algorithm"] == algorithm
    assert run.data.params["model.random_state"] == "42"

    if algorithm == "logistic_regression":
        assert "model.logistic_regression.C" in run.data.params
        assert "model.logistic_regression.max_iter" in run.data.params
        assert not any(k.startswith("model.random_forest") for k in run.data.params)
    else:
        assert "model.random_forest.n_estimators" in run.data.params
        assert "model.random_forest.max_depth" in run.data.params
        assert "model.random_forest.min_samples_leaf" in run.data.params
        assert not any(k.startswith("model.logistic_regression") for k in run.data.params)

    assert "data.test_size" in run.data.params
    assert "evaluation.primary_metric" in run.data.params
    assert "f1" in run.data.metrics

    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "reports" in artifacts

    model_info = mlflow.models.get_model_info(tracking_data["model_uri"])
    assert model_info.model_id == tracking_data["model_id"]
    loaded_model = mlflow.sklearn.load_model(tracking_data["model_uri"])
    assert hasattr(loaded_model, "predict")
