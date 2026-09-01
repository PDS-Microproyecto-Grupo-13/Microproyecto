from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pipeline.common.io import ensure_parent
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)


def preprocess(settings: Settings) -> None:
    source = settings.path("data/validated/dataset.csv")
    train_path = settings.path("data/processed/train.csv")
    test_path = settings.path("data/processed/test.csv")
    config = settings.section("data")
    LOGGER.info("preprocess | input=%s | outputs=%s,%s", source, train_path, test_path)
    frame = pd.read_csv(source)
    train, test = train_test_split(
        frame,
        test_size=float(config["test_size"]),
        random_state=int(config["random_state"]),
        stratify=frame["target"],
    )
    ensure_parent(train_path)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    LOGGER.info("preprocess | result=success | train_rows=%d test_rows=%d", len(train), len(test))
