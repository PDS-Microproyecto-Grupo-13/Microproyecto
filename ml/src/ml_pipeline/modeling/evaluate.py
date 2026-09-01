from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml_pipeline.common.io import write_json
from ml_pipeline.modeling.candidate import candidate_decision
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.lineage import collect_lineage

LOGGER = logging.getLogger(__name__)


def calculate_metrics(target: pd.Series, prediction: object) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
    }


def evaluate(settings: Settings) -> None:
    model_path = settings.path("artifacts/work/model/model.joblib")
    test_path = settings.path("data/processed/test.csv")
    LOGGER.info("evaluate | inputs=%s,%s", model_path, test_path)
    frame = pd.read_csv(test_path)
    model = joblib.load(model_path)
    metrics = calculate_metrics(frame["target"], model.predict(frame.drop(columns="target")))
    decision = candidate_decision(metrics, settings.section("evaluation"))
    evaluation = settings.section("evaluation")
    primary = str(evaluation["primary_metric"])
    manifest = {
        **collect_lineage(settings),
        "primary_metric": primary,
        "primary_metric_value": metrics[primary],
        "candidate": decision["eligible"],
    }
    write_json(settings.path("artifacts/reports/metrics.json"), metrics)
    write_json(settings.path("artifacts/reports/candidate.json"), decision)
    write_json(settings.path("artifacts/reports/experiment_manifest.json"), manifest)
    LOGGER.info("evaluate | result=success | %s=%.6f | eligible=%s", primary, metrics[primary], decision["eligible"])
