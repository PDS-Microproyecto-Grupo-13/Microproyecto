from pathlib import Path

import joblib
import pandas as pd

from ml_pipeline.data.collect import collect
from ml_pipeline.data.preprocess import preprocess
from ml_pipeline.data.validate import validate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings


def test_model_accepts_feature_schema_and_owns_inference_preprocessing(tmp_path: Path) -> None:
    (tmp_path / "params.yaml").write_text(
        """data:\n  test_size: 0.2\n  random_state: 42\nmodel:\n  algorithm: logistic_regression\n  max_iter: 1000\nevaluation:\n  primary_metric: f1\n  minimum_score: 0.8\n""",
        encoding="utf-8",
    )
    settings = Settings.load(tmp_path)
    for stage in (collect, validate, preprocess, train):
        stage(settings)
    model = joblib.load(tmp_path / "artifacts/work/model/model.joblib")
    sample = pd.read_csv(tmp_path / "data/processed/test.csv").drop(columns="target").head(2)
    prediction = model.predict(sample)
    assert list(model.named_steps) == ["imputer", "scaler", "classifier"]
    assert prediction.shape == (2,)
    assert set(prediction).issubset({0, 1})
