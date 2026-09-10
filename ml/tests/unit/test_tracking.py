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
from ml_pipeline.modeling.factory import build_model, effective_model_params
from ml_pipeline.tracking.mlflow_tracker import (
    effective_tracking_params,
    filter_inactive_algorithm_params,
    normalize_mlflow_param,
    track,
)


def test_normalize_mlflow_param() -> None:
    assert normalize_mlflow_param(None) == "null"
    assert normalize_mlflow_param(42) == 42
    assert normalize_mlflow_param(3.14) == 3.14
    assert normalize_mlflow_param("hello") == "hello"
    assert normalize_mlflow_param(True) is True
    assert normalize_mlflow_param(False) is False


def test_effective_tracking_params_random_forest_with_none_depth() -> None:
    params = {
        "model": {
            "algorithm": "random_forest",
            "random_state": 42,
            "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        }
    }
    effective = effective_tracking_params(params)
    assert effective["model.algorithm"] == "random_forest"
    assert effective["model.random_forest.max_depth"] == "null"
    assert effective["model.random_forest.n_estimators"] == 100


def test_consistency_factory_manifest_and_tracking() -> None:
    params = {
        "data": {"test_size": 0.2, "random_state": 42},
        "model": {
            "algorithm": "random_forest",
            "random_state": 42,
            "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 2},
        },
        "evaluation": {"primary_metric": "f1", "minimum_score": 0.8},
    }
    # 1. Factory
    factory_params = effective_model_params(params["model"])
    model = build_model(params["model"])
    rf = model.named_steps["classifier"]
    assert rf.max_depth is None
    assert rf.n_estimators == 100
    assert factory_params["max_depth"] is None

    # 2. Tracking
    tracking = effective_tracking_params(params)
    assert tracking["model.random_forest.max_depth"] == "null"
    assert tracking["model.random_forest.n_estimators"] == 100
    assert "model.logistic_regression.C" not in tracking


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


def test_effective_tracking_params_for_lightgbm() -> None:
    params = {
        "data": {
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "target_scope": "reportado",
        },
        "model": {
            "algorithm": "lightgbm",
            "random_state": 42,
            "lightgbm": {
                "objective": "regression_l1",
                "n_estimators": 700,
                "num_leaves": 95,
                "learning_rate": 0.06,
                "min_child_samples": 20,
            },
        },
        "qualification": {
            "min_improvement_vs_baseline": 0.10,
            "max_temporal_gap": 0.25,
            "uncertainty_quantile": 0.80,
        },
        "evaluation": {"primary_metric": "mae_promedio"},
    }

    effective = effective_tracking_params(params)
    assert effective["model.algorithm"] == "lightgbm"
    assert effective["model.random_state"] == 42
    assert effective["model.feature_count"] == 24
    assert effective["model.targets"] == "y_min_usd, y_max_usd"
    assert effective["model.lightgbm.n_estimators"] == 700
    assert effective["model.lightgbm.num_leaves"] == 95
    assert effective["model.lightgbm.learning_rate"] == 0.06
    assert effective["qualification.min_improvement_vs_baseline"] == 0.10
    assert effective["data.target_scope"] == "reportado"
    assert effective["evaluation.primary_metric"] == "mae_promedio"


def test_track_fails_if_model_or_metrics_missing(tmp_path: Path) -> None:
    """Verify track strictly raises FileNotFoundError if model.joblib or metrics.json are missing."""
    (tmp_path / "params.yaml").write_text("model:\n  algorithm: lightgbm\n", encoding="utf-8")
    settings = Settings.load(tmp_path)
    with pytest.raises(FileNotFoundError, match="Run pipeline stages through evaluate before track"):
        track(settings)


def test_track_creates_mlflow_run_with_pyfunc_params_metrics_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify track executes end-to-end on salary model without retraining or evaluating."""
    from unittest.mock import patch
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from ml_pipeline.features import FEATURE_COLUMNS
    from ml_pipeline.modeling.pyfunc import create_input_example, OUTPUT_COLUMNS

    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    experiment_name = "test-salary-tracking"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(experiment_name)

    # 1. Setup params.yaml
    (tmp_path / "params.yaml").write_text(
        """data:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  target_scope: reportado
model:
  algorithm: lightgbm
  random_state: 42
  lightgbm:
    objective: regression_l1
    n_estimators: 10
    num_leaves: 15
    learning_rate: 0.06
    min_child_samples: 2
    verbosity: -1
qualification:
  min_improvement_vs_baseline: 0.10
  max_temporal_gap: 0.25
  uncertainty_quantile: 0.80
evaluation:
  primary_metric: mae_promedio
""",
        encoding="utf-8",
    )

    # 2. Build mock bundle
    X_dummy = pd.DataFrame(0.0, index=range(10), columns=FEATURE_COLUMNS)
    for col in ["title", "country", "region", "experience_level", "work_mode", "company"]:
        X_dummy[col] = "known_val"

    pipe_min = Pipeline([
        ("preprocesamiento", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])),
        ("modelo", DummyRegressor(strategy="constant", constant=np.log1p(45000.0))),
    ])
    pipe_min.fit(X_dummy, np.log1p(np.array([45000.0] * 10)))

    pipe_max = Pipeline([
        ("preprocesamiento", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])),
        ("modelo", DummyRegressor(strategy="constant", constant=np.log1p(75000.0))),
    ])
    pipe_max.fit(X_dummy, np.log1p(np.array([75000.0] * 10)))

    bundle = {
        "pipeline_min": pipe_min,
        "pipeline_max": pipe_max,
        "train_limits": {"floor": 10935.57, "ceiling": 720000.0, "source_split": "train"},
        "uncertainty_margin": 50927.2876,
        "nominal_coverage": 0.80,
        "feature_columns": FEATURE_COLUMNS,
        "algorithm": "lightgbm",
        "dataset_fingerprint": "mock_fingerprint_fase6",
    }
    model_dir = tmp_path / "artifacts" / "work" / "model"
    model_dir.mkdir(parents=True)
    joblib.dump(bundle, model_dir / "model.joblib")

    # 3. Create mock evaluated reports from Phase 5
    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True)

    (reports_dir / "metrics.json").write_text(
        json.dumps({
            "mae_min": 22487.04,
            "mae_max": 31675.40,
            "mae_promedio": 27081.22,
            "rmse_min": 40174.00,
            "rmse_max": 54490.05,
            "mape_min": 0.2651,
            "mape_max": 0.2388,
            "r2_min": 0.5727,
            "r2_max": 0.6282,
            "mae_amplitud": 21127.06,
            "cobertura_intervalo": 0.1139,
            "incoherencia_raw": 0.0205,
            "prediccion_no_positiva": 0.0,
            "uncertainty_margin": 50927.2876,
            "uncertainty_nominal_coverage": 0.80,
            "uncertainty_test_coverage": 0.7893,
        }),
        encoding="utf-8",
    )
    (reports_dir / "qualification.json").write_text(
        json.dumps({
            "algorithm": "lightgbm",
            "eligible": True,
            "baseline_mae": 56034.61,
            "validation_mae": 26140.81,
            "improvement_vs_baseline": 0.5335,
            "temporal_gap": 0.0355,
        }),
        encoding="utf-8",
    )
    (reports_dir / "candidate.json").write_text(
        json.dumps({
            "algorithm": "lightgbm",
            "eligible": True,
            "dataset_fingerprint": "mock_fingerprint_fase6",
            "training": {
                "train_rows": 38054,
                "validation_rows": 8154,
                "final_training_rows": 46208,
            },
            "test_evaluation": {"test_rows": 8155},
        }),
        encoding="utf-8",
    )
    (reports_dir / "training.json").write_text(
        json.dumps({
            "algorithm": "lightgbm",
            "train_rows": 38054,
            "validation_rows": 8154,
            "final_training_rows": 46208,
        }),
        encoding="utf-8",
    )
    (reports_dir / "experiment_manifest.json").write_text(
        json.dumps({"algorithm": "lightgbm", "candidate": True}),
        encoding="utf-8",
    )

    settings = Settings.load(tmp_path)

    # 4. Verify track strictly does not call train or evaluate
    with patch("ml_pipeline.modeling.train.train") as mock_train, \
         patch("ml_pipeline.modeling.evaluate.evaluate") as mock_eval:
        run_id = track(settings)
        assert mock_train.call_count == 0
        assert mock_eval.call_count == 0

    assert isinstance(run_id, str) and len(run_id) > 0

    # 5. Verify tracking.json output report
    tracking_data = read_json(reports_dir / "tracking.json")
    assert tracking_data["run_id"] == run_id
    assert tracking_data["algorithm"] == "lightgbm"
    assert tracking_data["dataset_fingerprint"] == "mock_fingerprint_fase6"
    assert tracking_data["eligible"] is True
    assert "model_uri" in tracking_data

    # 6. Verify Run contents in MLflow
    run = client.get_run(run_id)
    assert run.info.run_name == "lightgbm"
    assert run.data.tags["algorithm"] == "lightgbm"
    assert run.data.tags["candidate_eligible"] == "True"
    assert run.data.params["model.algorithm"] == "lightgbm"
    assert run.data.params["model.feature_count"] == "24"
    assert run.data.params["model.targets"] == "y_min_usd, y_max_usd"
    assert run.data.params["model.lightgbm.n_estimators"] == "10"
    assert run.data.metrics["test_mae_promedio"] == 27081.22
    assert run.data.metrics["uncertainty_test_coverage"] == 0.7893

    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "reports" in artifacts

    model_info = mlflow.models.get_model_info(tracking_data["model_uri"])
    assert model_info.model_id == tracking_data["model_id"]

    # 7. Verify model can be loaded back via MLflow PyFunc API
    loaded_pyfunc = mlflow.pyfunc.load_model(tracking_data["model_uri"])
    test_input = create_input_example()
    preds = loaded_pyfunc.predict(test_input)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == OUTPUT_COLUMNS
    assert len(preds) == len(test_input)

    # 8. Verify Model Registry remains untouched (no registered versions)
    model_versions = client.search_model_versions("")
    assert len(model_versions) == 0

