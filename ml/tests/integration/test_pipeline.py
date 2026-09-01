from pathlib import Path

from ml_pipeline.common.io import read_json
from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings


def test_reproducible_pipeline_creates_expected_outputs(tmp_path: Path) -> None:
    (tmp_path / "params.yaml").write_text(
        """data:\n  test_size: 0.2\n  random_state: 42\nmodel:\n  algorithm: logistic_regression\n  max_iter: 1000\n  random_state: 42\nevaluation:\n  primary_metric: f1\n  minimum_score: 0.8\n""",
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
