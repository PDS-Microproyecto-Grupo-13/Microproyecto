from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)


def build_reference_model(config: dict[str, object]) -> Pipeline:
    algorithm = config.get("algorithm")
    if algorithm != "logistic_regression":
        raise ValueError(f"Unsupported reference algorithm: {algorithm!r}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(config["C"]),
                    max_iter=int(config["max_iter"]),
                    random_state=int(config.get("random_state", 42)),
                ),
            ),
        ]
    )


def train(settings: Settings) -> None:
    source = settings.path("data/processed/train.csv")
    output = settings.path("artifacts/work/model/model.joblib")
    LOGGER.info("train | input=%s | output=%s", source, output)
    frame = pd.read_csv(source)
    features = frame.drop(columns="target")
    model = build_reference_model(settings.section("model"))
    model.fit(features, frame["target"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    LOGGER.info("train | result=success | features=%d", features.shape[1])
