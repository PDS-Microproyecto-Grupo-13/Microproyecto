import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_pipeline.common.io import read_json
from ml_pipeline.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS
from ml_pipeline.modeling.evaluate import evaluate
from ml_pipeline.modeling.train import train
from ml_pipeline.settings import Settings


@pytest.fixture
def synthetic_eval_env(tmp_path: Path) -> Path:
    """Create an isolated test environment with train, val, test datasets, limits, and trained bundle."""
    rng = np.random.RandomState(42)
    n_train = 35
    n_val = 15
    n_test = 20

    def make_df(n: int, start_date: str) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, periods=n, freq="D")
        data: dict[str, Any] = {
            "id": list(range(1, n + 1)),
            "published": dates,
            "title": rng.choice(["Software Engineer", "Data Scientist", "Backend Lead"], size=n),
            "country": rng.choice(["spain", "germany", "desconocido"], size=n),
            "region": rng.choice(["europe", "americas", "desconocido"], size=n),
            "experience_level": rng.choice(["junior", "mid", "senior", "lead"], size=n),
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
    test_df = make_df(n_test, "2026-05-01")

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "validation.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

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
            "dataset_fingerprint": "mock-fingerprint-fase5",
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
            "dataset_fingerprint": "mock-fingerprint-fase5",
            "baseline_mae": 56000.0,
            "validation_mae": 26000.0,
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
evaluation:
  primary_metric: mae_promedio
""",
        encoding="utf-8",
    )

    # Train model to generate model.joblib and training.json
    settings = Settings.load(tmp_path)
    train(settings)

    return tmp_path


def test_evaluate_does_not_retrain_or_fit(synthetic_eval_env: Path) -> None:
    """Verify evaluate strictly loads existing model and NEVER calls fit/retrain."""
    settings = Settings.load(synthetic_eval_env)

    with patch.object(Pipeline, "fit") as mock_fit:
        evaluate(settings)
        assert mock_fit.call_count == 0


def test_evaluate_consumes_exact_bundle_and_does_not_alter_limits_or_uncertainty(synthetic_eval_env: Path) -> None:
    """Verify evaluate preserves exact train_limits and uncertainty_margin from Phase 3/4."""
    settings = Settings.load(synthetic_eval_env)
    candidate = evaluate(settings)

    metrics_path = synthetic_eval_env / "artifacts/reports/metrics.json"
    metrics = read_json(metrics_path)

    assert metrics["uncertainty_margin"] == 50927.2876
    assert metrics["uncertainty_nominal_coverage"] == 0.80
    assert 0.0 <= metrics["uncertainty_test_coverage"] <= 1.0

    bundle = joblib.load(synthetic_eval_env / "artifacts/work/model/model.joblib")
    assert bundle["train_limits"]["floor"] == 10935.571999999996
    assert bundle["train_limits"]["ceiling"] == 720000.0
    assert bundle["uncertainty_margin"] == 50927.2876


def test_evaluate_regression_metrics_and_invariants(synthetic_eval_env: Path) -> None:
    """Verify regression metrics and quality checks are computed and hold true on test."""
    settings = Settings.load(synthetic_eval_env)
    evaluate(settings)

    metrics = read_json(synthetic_eval_env / "artifacts/reports/metrics.json")

    # Standard metrics presence
    for key in ["mae_min", "mae_max", "mae_promedio", "rmse_min", "rmse_max", "mape_min", "mape_max", "r2_min", "r2_max", "mae_amplitud", "cobertura_intervalo", "incoherencia_raw", "prediccion_no_positiva"]:
        assert key in metrics
        assert isinstance(metrics[key], float)

    # MAE promedio definition
    expected_mean_mae = (metrics["mae_min"] + metrics["mae_max"]) / 2.0
    assert abs(metrics["mae_promedio"] - expected_mean_mae) < 1e-4

    # Quality checks (invariants)
    qc = metrics["quality_checks"]
    assert qc["valores_finitos"] is True
    assert qc["valores_positivos"] is True
    assert qc["rangos_ordenados"] is True
    assert qc["dentro_limites_operativos"] is True

    # Diagnostic checks
    diag = metrics["diagnostic_checks"]
    assert "nuevo_perfil_valido" in diag
    assert "progresion_experiencia_no_decreciente" in diag


def test_evaluate_candidate_report_preserves_qualification(synthetic_eval_env: Path) -> None:
    """Verify candidate.json preserves qualification approval and records test results."""
    settings = Settings.load(synthetic_eval_env)
    candidate = evaluate(settings)

    candidate_file = read_json(synthetic_eval_env / "artifacts/reports/candidate.json")

    assert candidate_file["eligible"] is True
    assert candidate_file["qualification"]["eligible"] is True
    assert candidate_file["test_evaluation"]["test_rows"] == 20
    assert candidate_file["test_evaluation"]["primary_metric"] == "mae_promedio"
    assert candidate_file["test_evaluation"]["primary_metric_value"] > 0.0


def test_evaluate_generates_all_deterministic_reports_without_timestamps(synthetic_eval_env: Path) -> None:
    """Verify all 7 reports are created without any dynamic timestamps."""
    settings = Settings.load(synthetic_eval_env)
    evaluate(settings)

    reports = [
        "metrics.json",
        "candidate.json",
        "experiment_manifest.json",
        "audit_segments.json",
        "audit_novelty.json",
        "audit_sensitivity.json",
        "feature_importance.json",
    ]

    for rep in reports:
        rep_path = synthetic_eval_env / "artifacts/reports" / rep
        assert rep_path.is_file(), f"Missing expected report: {rep}"
        content = rep_path.read_text(encoding="utf-8")
        assert "timestamp" not in content.lower()
        assert "created_at" not in content.lower()
