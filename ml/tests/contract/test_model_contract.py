from pathlib import Path

import joblib
import pandas as pd

import pytest
from sklearn.pipeline import Pipeline

from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings


@pytest.mark.skip(reason="Model contract belongs to post-data migration phases")
@pytest.mark.parametrize(
    "algorithm, expected_steps",
    [
        ("logistic_regression", ["imputer", "scaler", "classifier"]),
        ("random_forest", ["imputer", "classifier"]),
    ],
)
def test_model_accepts_feature_schema_and_owns_inference_preprocessing(
    tmp_path: Path, algorithm: str, expected_steps: list[str]
) -> None:
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
    for stage in (collect, validate, preprocess, train):
        stage(settings)
    model = joblib.load(tmp_path / "artifacts/work/model/model.joblib")
    sample = pd.read_csv(tmp_path / "data/processed/test.csv").drop(columns="target").head(2)
    prediction = model.predict(sample)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == expected_steps
    assert prediction.shape == (2,)
    assert set(prediction).issubset({0, 1})
