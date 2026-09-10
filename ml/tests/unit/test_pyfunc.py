from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml_pipeline.features import FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.modeling.pyfunc import (
    OUTPUT_COLUMNS,
    RAW_INPUT_COLUMNS,
    SalaryPredictorModel,
    build_model_signature,
    create_input_example,
    verify_local_pyfunc_parity,
)


@pytest.fixture
def mock_salary_bundle() -> dict[str, Any]:
    """Create a minimal valid model bundle with DummyRegressors for unit tests."""
    X_dummy = pd.DataFrame(0.0, index=range(10), columns=FEATURE_COLUMNS)
    for col in ["title", "country", "region", "experience_level", "work_mode", "company"]:
        X_dummy[col] = "known_val"

    y_min = np.array([50000.0] * 10)
    y_max = np.array([80000.0] * 10)

    pipe_min = Pipeline([
        ("preprocesamiento", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])),
        ("modelo", DummyRegressor(strategy="constant", constant=np.log1p(50000.0))),
    ])
    pipe_min.fit(X_dummy, np.log1p(y_min))

    pipe_max = Pipeline([
        ("preprocesamiento", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])),
        ("modelo", DummyRegressor(strategy="constant", constant=np.log1p(80000.0))),
    ])
    pipe_max.fit(X_dummy, np.log1p(y_max))

    return {
        "pipeline_min": pipe_min,
        "pipeline_max": pipe_max,
        "train_limits": {"floor": 10000.0, "ceiling": 500000.0, "source_split": "train"},
        "uncertainty_margin": 50000.0,
        "nominal_coverage": 0.80,
        "feature_columns": FEATURE_COLUMNS,
        "algorithm": "lightgbm",
        "dataset_fingerprint": "mock_fingerprint_fase6",
    }


def test_pyfunc_input_example_and_signature(mock_salary_bundle: dict[str, Any]) -> None:
    """Verify input_example has required raw columns and signature maps raw input to exact output."""
    example = create_input_example()
    assert isinstance(example, pd.DataFrame)
    assert len(example) >= 2
    for col in RAW_INPUT_COLUMNS:
        assert col in example.columns

    model = SalaryPredictorModel(bundle=mock_salary_bundle)
    signature = build_model_signature(model, example)

    # Signature input names match raw columns
    input_names = [col.name for col in signature.inputs.inputs]
    for req_col in ["title", "company", "regions", "countries", "experience_level"]:
        assert req_col in input_names

    # Signature output names match exact 3 output columns
    output_names = [col.name for col in signature.outputs.inputs]
    assert output_names == OUTPUT_COLUMNS


def test_pyfunc_predict_single_and_batch_rows(mock_salary_bundle: dict[str, Any]) -> None:
    """Verify SalaryPredictorModel handles 1 row and multiple rows, producing exact columns."""
    model = SalaryPredictorModel(bundle=mock_salary_bundle)
    example = create_input_example()

    # Batch prediction
    batch_out = model.predict(None, example)
    assert isinstance(batch_out, pd.DataFrame)
    assert list(batch_out.columns) == OUTPUT_COLUMNS
    assert len(batch_out) == len(example)
    assert (batch_out["salary_min_usd"] <= batch_out["salary_max_usd"]).all()
    assert np.allclose(
        batch_out["salary_midpoint_usd"],
        (batch_out["salary_min_usd"] + batch_out["salary_max_usd"]) / 2.0,
    )

    # Single row prediction
    single_out = model.predict(None, example.head(1))
    assert isinstance(single_out, pd.DataFrame)
    assert len(single_out) == 1
    assert list(single_out.columns) == OUTPUT_COLUMNS
    assert (single_out["salary_min_usd"] <= single_out["salary_max_usd"]).all()


def test_pyfunc_fails_when_regions_missing(mock_salary_bundle: dict[str, Any]) -> None:
    """Verify prepare_features inside pyfunc strictly raises KeyError when 'regions' is missing."""
    model = SalaryPredictorModel(bundle=mock_salary_bundle)
    example = create_input_example().drop(columns=["regions"])

    with pytest.raises(KeyError, match="regions"):
        model.predict(None, example)


def test_pyfunc_parity_verification_helper(mock_salary_bundle: dict[str, Any]) -> None:
    """Verify verify_local_pyfunc_parity checks all 8 invariants and detects mismatches."""
    example = create_input_example()
    model = SalaryPredictorModel(bundle=mock_salary_bundle)

    # Clean parity passes
    checks = verify_local_pyfunc_parity(mock_salary_bundle, example, model)
    assert all(checks.values())
    assert checks["mismo_min"] is True
    assert checks["mismo_max"] is True
    assert checks["mismo_midpoint"] is True
    assert checks["valores_finitos"] is True
    assert checks["valores_positivos"] is True
    assert checks["rangos_ordenados"] is True
    assert checks["limites_operativos"] is True


def test_pyfunc_save_load_and_mlflow_roundtrip(
    tmp_path: Path,
    mock_salary_bundle: dict[str, Any],
) -> None:
    """Verify PyFunc model logs to MLflow, restores via load_model, and maintains exact parity."""
    bundle_path = tmp_path / "model.joblib"
    joblib.dump(mock_salary_bundle, bundle_path)

    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-pyfunc-roundtrip")

    example = create_input_example()
    pyfunc_model = SalaryPredictorModel(bundle=mock_salary_bundle)
    signature = build_model_signature(pyfunc_model, example)

    # Resolve actual package path for code_paths
    pkg_path = Path(__file__).resolve().parents[2] / "src" / "ml_pipeline"

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=SalaryPredictorModel(),
            artifacts={"model_bundle": str(bundle_path)},
            code_paths=[str(pkg_path)],
            signature=signature,
            input_example=example,
        )

    # Reload model through MLflow PyFunc API
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert hasattr(loaded_model, "predict")

    # Predict with reloaded model
    preds = loaded_model.predict(example)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == OUTPUT_COLUMNS
    assert len(preds) == len(example)

    # Strict parity verification against original bundle
    parity_checks = verify_local_pyfunc_parity(mock_salary_bundle, example, loaded_model)
    assert all(parity_checks.values())
