from __future__ import annotations

import logging

from sklearn.datasets import load_breast_cancer

from ml_pipeline.common.io import ensure_parent
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)


def collect(settings: Settings) -> None:
    output = settings.path("data/raw/dataset.csv")
    LOGGER.info("collect | source=sklearn:breast_cancer | output=%s", output)
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.rename(columns={"target": "target"})
    ensure_parent(output)
    frame.to_csv(output, index=False)
    LOGGER.info("collect | result=success | rows=%d columns=%d", *frame.shape)
