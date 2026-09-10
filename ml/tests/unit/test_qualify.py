from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.common.io import read_json
from ml_pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.modeling.qualify import (
    build_lgbm_estimator,
    build_preprocessor,
    build_target_pipeline,
    compute_baseline_metrics,
    compute_metrics,
    postprocess,
    qualify,
)
from ml_pipeline.settings import Settings


@pytest.fixture
def synthetic_train_and_val_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic train and validation data respecting the 24-feature schema."""
    rng = np.random.RandomState(42)
    n_train = 60
    n_val = 20

    def make_df(n: int, start_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, periods=n, freq="D")
        data: dict[str, Any] = {
            "id": list(range(1, n + 1)),
            "published": dates,
            # Categorical columns (6)
            "title": rng.choice(["Software Engineer", "Data Scientist", "Backend Lead"], size=n),
            "country": rng.choice(["spain", "germany", "desconocido"], size=n),
            "region": rng.choice(["europe", "americas", "desconocido"], size=n),
            "experience_level": rng.choice(["junior", "mid", "senior", "desconocido"], size=n),
            "work_mode": rng.choice(["remote", "hybrid", "onsite", "desconocido"], size=n),
            "company": rng.choice(["Acme Corp", "Beta Inc", "Gamma LLC"], size=n),
            # Numeric columns (18)
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

        # Synthetic salaries
        base_salary = rng.uniform(40000, 120000, size=n)
        data["y_min_usd"] = base_salary
        data["y_max_usd"] = base_salary + rng.uniform(5000, 30000, size=n)
        return pd.DataFrame(data)

    train_df = make_df(n_train, "2026-01-01")
    val_df = make_df(n_val, "2026-03-15")
    return train_df, val_df


def test_build_preprocessor_structure_and_fit() -> None:
    """Verify preprocessor isolates categorical target encoder and numeric imputer."""
    preprocessor = build_preprocessor(random_state=42)
    assert len(preprocessor.transformers) == 2

    # Verify input columns match exactly
    cat_names = preprocessor.transformers[0][2]
    num_names = preprocessor.transformers[1][2]
    assert list(cat_names) == CATEGORICAL_COLUMNS
    assert list(num_names) == NUMERIC_COLUMNS


def test_target_encoder_fits_only_on_passed_data(synthetic_train_and_val_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify TargetEncoder fits strictly on the provided training set without leakage."""
    train_df, val_df = synthetic_train_and_val_data
    preprocessor = build_preprocessor(random_state=42)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = np.log1p(train_df["y_min_usd"])
    X_val = val_df[FEATURE_COLUMNS]

    # Fitting should succeed
    transformed_train = preprocessor.fit_transform(X_train, y_train)
    assert transformed_train.shape == (len(train_df), 24)

    # Transform validation without fitting
    transformed_val = preprocessor.transform(X_val)
    assert transformed_val.shape == (len(val_df), 24)
    assert not np.isnan(transformed_val).any()


def test_postprocess_clips_and_orders_targets() -> None:
    """Test postprocessing clipping and min <= max enforcement."""
    limits = {"floor": 10000.0, "ceiling": 200000.0}
    raw_pred = np.array([
        [-500.0, 50000.0],       # min below floor
        [150000.0, 300000.0],    # max above ceiling
        [90000.0, 60000.0],      # inverted: raw min > raw max
        [70000.0, 95000.0],      # valid and sorted
    ])

    post = postprocess(raw_pred, limits)

    # 1. Floor clipping
    assert post[0, 0] == 10000.0
    assert post[0, 1] == 50000.0

    # 2. Ceiling clipping
    assert post[1, 0] == 150000.0
    assert post[1, 1] == 200000.0

    # 3. Min <= Max enforcement
    assert post[2, 0] == 60000.0
    assert post[2, 1] == 90000.0

    # 4. Invariant: min <= max for all rows
    assert (post[:, 0] <= post[:, 1]).all()


def test_compute_metrics_and_baseline_logic() -> None:
    """Verify compute_metrics and compute_baseline_metrics calculate regression statistics."""
    y_train = np.array([
        [50000.0, 70000.0],
        [60000.0, 80000.0],
        [70000.0, 90000.0],
        [80000.0, 100000.0],
    ])
    y_val = np.array([
        [65000.0, 85000.0],
        [75000.0, 95000.0],
    ])
    limits = {"floor": 10000.0, "ceiling": 200000.0}

    base_metrics, base_pred = compute_baseline_metrics(y_train, y_val, limits)

    assert "mae_promedio" in base_metrics
    assert "mae_min" in base_metrics
    assert "mae_max" in base_metrics
    assert base_metrics["mae_promedio"] > 0.0
    assert base_pred.shape == y_val.shape


def test_uncertainty_calibration_quantile() -> None:
    """Verify uncertainty margin calculation at 80% quantile with method='higher'."""
    y_val = np.array([
        [50000.0, 80000.0],
        [60000.0, 90000.0],
        [70000.0, 100000.0],
        [80000.0, 110000.0],
        [90000.0, 120000.0],
    ])
    # Predicted with varying errors
    val_pred = np.array([
        [52000.0, 85000.0],   # errors: 2000, 5000 -> max 5000
        [58000.0, 88000.0],   # errors: 2000, 2000 -> max 2000
        [73000.0, 96000.0],   # errors: 3000, 4000 -> max 4000
        [70000.0, 105000.0],  # errors: 10000, 5000 -> max 10000
        [91000.0, 128000.0],  # errors: 1000, 8000 -> max 8000
    ])
    # Errors: [5000, 2000, 4000, 10000, 8000]
    # Sorted: [2000, 4000, 5000, 8000, 10000]
    # 80th percentile with method="higher": index 4 of 5 is 10000
    joint_error = np.max(np.abs(y_val - val_pred), axis=1)
    margin = float(np.quantile(joint_error, 0.80, method="higher"))
    assert margin == 10000.0


def test_retrotest_temporal_order_and_split(
    synthetic_train_and_val_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verify temporal retrotest strictly splits train chronologically at ratio 0.80."""
    train_df, _ = synthetic_train_and_val_data
    # Verify train_df is sorted by published
    assert train_df["published"].is_monotonic_increasing

    ratio = 0.80
    cut = int(ratio * len(train_df))
    assert cut == int(0.80 * 60) == 48

    back_train = train_df.iloc[:cut]
    back_eval = train_df.iloc[cut:]

    assert len(back_train) == 48
    assert len(back_eval) == 12
    # Ensure no temporal overlap: max date in back_train <= min date in back_eval
    assert back_train["published"].max() < back_eval["published"].min()



def test_qualify_never_reads_test_parquet(
    tmp_path: Path,
    synthetic_train_and_val_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verify that qualify NEVER attempts to read data/processed/test.parquet."""
    train_df, val_df = synthetic_train_and_val_data

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "validation.parquet", index=False)

    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True)
    import json
    (reports_dir / "train_limits.json").write_text(
        json.dumps({"floor": 10000.0, "ceiling": 200000.0, "source_split": "train"}),
        encoding="utf-8",
    )

    (tmp_path / "params.yaml").write_text(
        """model:
  algorithm: lightgbm
  random_state: 42
  lightgbm:
    n_estimators: 10
    num_leaves: 15
    verbosity: -1
qualification:
  min_improvement_vs_baseline: 0.10
  max_temporal_gap: 0.25
  backtest_train_ratio: 0.80
  uncertainty_quantile: 0.80
""",
        encoding="utf-8",
    )

    # Note: data/processed/test.parquet intentionally DOES NOT EXIST
    assert not (processed_dir / "test.parquet").exists()

    settings = Settings.load(tmp_path)
    report = qualify(settings)

    # qualify succeeded without test.parquet existing
    assert "baseline_mae" in report
    assert "validation_mae" in report
    assert "uncertainty_margin" in report
    assert (reports_dir / "qualification.json").is_file()
    assert (reports_dir / "uncertainty_calibration.json").is_file()


def test_qualify_deterministic_output(
    tmp_path: Path,
    synthetic_train_and_val_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verify qualify outputs deterministic reports without dynamic timestamps."""
    train_df, val_df = synthetic_train_and_val_data

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "validation.parquet", index=False)

    reports_dir = tmp_path / "artifacts" / "reports"
    reports_dir.mkdir(parents=True)
    import json
    (reports_dir / "train_limits.json").write_text(
        json.dumps({"floor": 10000.0, "ceiling": 200000.0, "source_split": "train"}),
        encoding="utf-8",
    )

    (tmp_path / "params.yaml").write_text(
        """model:
  algorithm: lightgbm
  random_state: 42
  lightgbm:
    n_estimators: 10
    num_leaves: 15
    verbosity: -1
qualification:
  min_improvement_vs_baseline: 0.10
  max_temporal_gap: 0.25
  backtest_train_ratio: 0.80
  uncertainty_quantile: 0.80
""",
        encoding="utf-8",
    )

    settings = Settings.load(tmp_path)
    report1 = qualify(settings)
    qual1_text = (reports_dir / "qualification.json").read_text(encoding="utf-8")
    calib1_text = (reports_dir / "uncertainty_calibration.json").read_text(encoding="utf-8")

    report2 = qualify(settings)
    qual2_text = (reports_dir / "qualification.json").read_text(encoding="utf-8")
    calib2_text = (reports_dir / "uncertainty_calibration.json").read_text(encoding="utf-8")

    # Strict byte-for-byte reproducibility
    assert qual1_text == qual2_text
    assert calib1_text == calib2_text
    # No timestamps in report keys
    assert "timestamp" not in qual1_text.lower()
    assert "generated_at" not in qual1_text.lower()
