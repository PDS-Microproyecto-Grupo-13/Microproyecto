from __future__ import annotations

import logging

import joblib
import pandas as pd

from ml_pipeline.modeling.factory import build_model, build_reference_model
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)

__all__ = ["build_model", "build_reference_model", "train"]


def train(settings: Settings) -> None:
    source = settings.path("data/processed/train.csv")
    output = settings.path("artifacts/work/model/model.joblib")
    model_config = settings.section("model")
    algorithm = model_config.get("algorithm")
    LOGGER.info("train | algorithm=%s | input=%s | output=%s", algorithm, source, output)
    frame = pd.read_csv(source)
    features = frame.drop(columns="target")
    model = build_model(model_config)
    model.fit(features, frame["target"])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    LOGGER.info(
        "train | result=success | algorithm=%s | features=%d",
        algorithm,
        features.shape[1],
    )
