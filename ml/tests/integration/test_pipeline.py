from pathlib import Path

import pytest

from ml_pipeline.common.io import read_json
from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings


@pytest.mark.parametrize("algorithm", ["logistic_regression", "random_forest"])
def test_reproducible_pipeline_creates_expected_outputs(tmp_path: Path, algorithm: str) -> None:
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

    expected = [
        "data/raw/dataset.csv",
        "data/validated/dataset.csv",
        "data/processed/train.csv",
        "data/processed/test.csv",
        "artifacts/work/model/model.joblib",
        "artifacts/reports/validation.json",
        "artifacts/reports/metrics.json",
        "artifacts/reports/candidate.json",
        "artifacts/reports/experiment_manifest.json",
    ]
    assert all((tmp_path / name).is_file() for name in expected)
    assert read_json(tmp_path / "artifacts/reports/validation.json")["valid"] is True
    assert read_json(tmp_path / "artifacts/reports/candidate.json")["eligible"] is True
    manifest = read_json(tmp_path / "artifacts/reports/experiment_manifest.json")
    assert manifest["algorithm"] == algorithm
    assert manifest["model"]["algorithm"] == algorithm
    assert isinstance(manifest["model"]["parameters"], dict)
    assert manifest["primary_metric"] == "f1"
    assert isinstance(manifest["primary_metric_value"], float)
    assert manifest["candidate"] is True
    if algorithm == "logistic_regression":
        assert "C" in manifest["model"]["parameters"]
        assert "max_iter" in manifest["model"]["parameters"]
    elif algorithm == "random_forest":
        assert "n_estimators" in manifest["model"]["parameters"]
        assert "max_depth" in manifest["model"]["parameters"]
