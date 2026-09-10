import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from ml_pipeline.common.io import read_json
from ml_pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.modeling.train import (
    TARGET_COLUMNS,
    predict_range,
    predict_with_uncertainty,
    train,
)
from ml_pipeline.settings import Settings


@pytest.fixture
def synthetic_train_env(tmp_path: Path) -> Path:
    """Create an isolated environment with small synthetic train/val datasets and Phase 3 reports."""
    rng = np.random.RandomState(42)
    n_train = 35
    n_val = 15

    def make_df(n: int, start_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, periods=n, freq="D")
        data: dict[str, Any] = {
            "id": list(range(1, n + 1)),
            "published": dates,
            "title": rng.choice(["Software Engineer", "Data Scientist", "Backend Lead"], size=n),
            "country": rng.choice(["spain", "germany", "desconocido"], size=n),
            "region": rng.choice(["europe", "americas", "desconocido"], size=n),
            "experience_level": rng.choice(["junior", "mid", "senior", "desconocido"], size=n),
            "work_mode": rng.choice(["remote", "hybrid", "onsite", "desconocido"], size=n),
            "company": rng.choice(["Acme Corp", "Beta Inc", "Gamma LLC"], size=n),
            "company_is_agency": rng.choice([0, 1], size=n),
            "experience_years": rng.choice([1.0, 3.0, 5.0, np.nan], size=n),
            "experience_years_missing": rng.choice([0, 1], size=n),
            "published_year": [d.year for d in dates],
            "published_month": [d.month for d in dates],
        }
        for skill in [
            "python", "sql", "aws", "azure", "gcp", "spark",
            "docker", "kubernetes", "machine_learning", "pytorch",
            "tensorflow", "tableau", "power_bi",
        ]:
            data[f"skill_{skill}"] = rng.choice([0, 1], size=n)

        base_salary = rng.uniform(40000, 120000, size=n)
        data["y_min_usd"] = base_salary
        data["y_max_usd"] = base_salary + rng.uniform(5000, 30000, size=n)
        return pd.DataFrame(data)

    train_df = make_df(n_train, "2026-01-01")
    val_df = make_df(n_val, "2026-03-01")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "validation.parquet", index=False)

    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True)

    (reports_dir / "train_limits.json").write_text(
        json.dumps({
            "floor": 10935.571999999996,
            "ceiling": 720000.0,
            "source_split": "train",
        }),
        encoding="utf-8",
    )
    (reports_dir / "uncertainty_calibration.json").write_text(
        json.dumps({
            "dataset_fingerprint": "mock-fingerprint-12345",
            "nominal_coverage": 0.80,
            "quantile_method": "higher",
            "source_split": "validation",
            "uncertainty_margin": 50927.2876,
            "validation_rows": n_val,
        }),
        encoding="utf-8",
    )
    (reports_dir / "qualification.json").write_text(
        json.dumps({
            "algorithm": "lightgbm",
            "eligible": True,
            "reasons": [],
            "uncertainty_margin": 50927.2876,
            "dataset_fingerprint": "mock-fingerprint-12345",
        }),
        encoding="utf-8",
    )

    (tmp_path / "params.yaml").write_text(
        """model:
  algorithm: lightgbm
  random_state: 42
  lightgbm:
    n_estimators: 10
    num_leaves: 15
    min_child_samples: 2
    verbosity: -1
""",
        encoding="utf-8",
    )

    return tmp_path


def test_train_fails_if_qualification_not_eligible(synthetic_train_env: Path) -> None:
    """Verify training aborts immediately if qualification eligible is false or missing."""
    qual_path = synthetic_train_env / "artifacts/reports/qualification.json"

    # Case 1: eligible is False
    qual_data = read_json(qual_path)
    qual_data["eligible"] = False
    qual_data["reasons"] = ["improvement_vs_baseline below threshold"]
    qual_path.write_text(json.dumps(qual_data), encoding="utf-8")

    settings = Settings.load(synthetic_train_env)
    with pytest.raises(RuntimeError, match="Training gate failed.*not eligible"):
        train(settings)

    # Case 2: qualification report does not exist
    qual_path.unlink()
    with pytest.raises(RuntimeError, match="qualification report not found"):
        train(settings)


def test_final_fit_uses_exactly_train_plus_validation_and_never_reads_test(synthetic_train_env: Path) -> None:
    """Verify training concatenates exactly train + val and succeeds without test.parquet."""
    settings = Settings.load(synthetic_train_env)

    # Ensure test.parquet does NOT exist
    test_parquet = synthetic_train_env / "data/processed/test.parquet"
    assert not test_parquet.exists()

    report = train(settings)

    assert report["train_rows"] == 35
    assert report["validation_rows"] == 15
    assert report["final_training_rows"] == 50
    assert report["qualification"]["eligible"] is True
    assert (synthetic_train_env / "artifacts/work/model/model.joblib").is_file()
    assert (synthetic_train_env / "artifacts/reports/training.json").is_file()


def test_bundle_structure_and_target_encoder_isolation(synthetic_train_env: Path) -> None:
    """Verify model.joblib bundle contents, dual pipelines and embedded TargetEncoders."""
    settings = Settings.load(synthetic_train_env)
    train(settings)

    bundle_path = synthetic_train_env / "artifacts/work/model/model.joblib"
    bundle = joblib.load(bundle_path)

    assert "pipeline_min" in bundle
    assert "pipeline_max" in bundle
    assert bundle["algorithm"] == "lightgbm"
    assert bundle["train_limits"]["floor"] == 10935.571999999996
    assert bundle["train_limits"]["ceiling"] == 720000.0
    assert bundle["uncertainty_margin"] == 50927.2876
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    assert bundle["targets"] == ["y_min_usd", "y_max_usd"]

    # Verify each pipeline has ColumnTransformer with TargetEncoder for categoricals
    for key in ["pipeline_min", "pipeline_max"]:
        pipe: Pipeline = bundle[key]
        preprocessor = pipe.named_steps["preprocesamiento"]
        # Find categorical transformer
        cat_step = preprocessor.transformers_[0]
        cat_pipeline = cat_step[1]
        target_encoder = cat_pipeline.named_steps["codificar"]
        assert isinstance(target_encoder, TargetEncoder)


def test_predict_range_and_uncertainty_invariants(synthetic_train_env: Path) -> None:
    """Test inference functions guarantee finite, positive, sorted and bounded ranges."""
    settings = Settings.load(synthetic_train_env)
    train(settings)

    bundle = joblib.load(synthetic_train_env / "artifacts/work/model/model.joblib")
    val_df = pd.read_parquet(synthetic_train_env / "data/processed/validation.parquet")

    # 1. predict_range
    preds = predict_range(val_df, bundle)

    assert preds.shape == (len(val_df), 2)
    assert np.isfinite(preds).all()
    assert (preds > 0).all()
    assert (preds[:, 0] <= preds[:, 1]).all()
    assert (preds[:, 0] >= bundle["train_limits"]["floor"]).all()
    assert (preds[:, 1] <= bundle["train_limits"]["ceiling"]).all()

    # 2. predict_with_uncertainty
    point_pred, lower, upper = predict_with_uncertainty(val_df, bundle)

    np.testing.assert_array_equal(point_pred, preds)
    assert (lower <= point_pred).all()
    assert (point_pred <= upper).all()
    assert (lower >= bundle["train_limits"]["floor"]).all()
    assert (upper <= bundle["train_limits"]["ceiling"]).all()


def test_serialize_and_load_preserves_predictions(synthetic_train_env: Path) -> None:
    """Verify serialized model bundle preserves exact byte/numerical predictions when reloaded."""
    settings = Settings.load(synthetic_train_env)
    train(settings)

    model_path = synthetic_train_env / "artifacts/work/model/model.joblib"
    bundle1 = joblib.load(model_path)

    val_df = pd.read_parquet(synthetic_train_env / "data/processed/validation.parquet")
    preds1 = predict_range(val_df, bundle1)

    # Reload bundle independently
    bundle2 = joblib.load(model_path)
    preds2 = predict_range(val_df, bundle2)

    np.testing.assert_array_equal(preds1, preds2)


def test_deterministic_training_report_no_timestamps(synthetic_train_env: Path) -> None:
    """Verify training report is deterministic without dynamic timestamps."""
    settings = Settings.load(synthetic_train_env)
    train(settings)

    report_path = synthetic_train_env / "artifacts/reports/training.json"
    content = report_path.read_text(encoding="utf-8")
    report = json.loads(content)

    assert "timestamp" not in content.lower()
    assert "created_at" not in content.lower()
    assert report["algorithm"] == "lightgbm"
    assert report["final_training_rows"] == 50
    assert report["feature_count"] == 24
    assert report["qualification"]["eligible"] is True
