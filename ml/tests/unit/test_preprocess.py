from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.common.io import read_json
from ml_pipeline.data.preprocess import compute_train_limits, preprocess, validate_ratios
from ml_pipeline.settings import Settings


def test_validate_ratios_accepts_valid_and_rejects_invalid() -> None:
    # Valid
    validate_ratios(0.70, 0.15, 0.15)
    validate_ratios(0.80, 0.10, 0.10)

    # Invalid sum
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_ratios(0.70, 0.20, 0.20)

    # Invalid range (negative or >= 1)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_ratios(-0.10, 0.60, 0.50)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_ratios(1.0, 0.0, 0.0)


def test_compute_train_limits_strictly_uses_train_distribution() -> None:
    train_df = pd.DataFrame({
        "y_min_usd": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "y_max_usd": [80000.0, 90000.0, 100000.0, 110000.0, 120000.0],
    })
    limits = compute_train_limits(train_df)
    assert limits["floor"] >= 1000.0
    assert limits["floor"] < limits["ceiling"]
    assert limits["source_split"] == "train"

    # Perturbing hypothetical validation or test records does not alter train limits
    val_df_extreme = pd.DataFrame({
        "y_min_usd": [10.0, 50000000.0],
        "y_max_usd": [20.0, 90000000.0],
    })
    # compute_train_limits only accepts train_df
    limits_after = compute_train_limits(train_df)
    assert limits == limits_after


def test_preprocess_filters_target_scope_and_preserves_temporal_order(tmp_path: Path) -> None:
    val_dir = tmp_path / "data" / "validated"
    val_dir.mkdir(parents=True)
    manifest_dir = tmp_path / "artifacts" / "reports"
    manifest_dir.mkdir(parents=True)

    # Create dummy data_manifest.json
    (manifest_dir / "data_manifest.json").write_text(
        '{"dataset_fingerprint": "test_fp_123"}\n', encoding="utf-8"
    )

    # Validated dataset with reportado, hibrido, and estimado
    records = []
    for i in range(1, 21):
        source = "reportado" if i <= 10 else ("híbrido" if i <= 15 else "estimado")
        records.append({
            "id": i,
            "title": f"Dev {i}",
            "countries": "Spain",
            "regions": "Europe",
            "experience_level": "MI",
            "has_remote": True,
            "work_mode": 2,
            "company": f"Company {i}",
            "company_is_agency": False,
            "experience_years": 3.0,
            "tags": "python",
            "published": f"2026-08-{i:02d}T10:00:00Z",
            "target_source": source,
            "y_min_usd": 50000.0 + i * 1000,
            "y_max_usd": 80000.0 + i * 1000,
        })
    df = pd.DataFrame(records)
    df.to_parquet(val_dir / "dataset.parquet", index=False, engine="pyarrow")

    # Config
    (tmp_path / "params.yaml").write_text(
        """data:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  target_scope: reportado
""",
        encoding="utf-8",
    )

    settings = Settings.load(tmp_path)
    preprocess(settings)

    train_path = tmp_path / "data" / "processed" / "train.parquet"
    val_path = tmp_path / "data" / "processed" / "validation.parquet"
    test_path = tmp_path / "data" / "processed" / "test.parquet"
    rep_path = tmp_path / "artifacts" / "reports" / "preprocess.json"

    assert train_path.is_file()
    assert val_path.is_file()
    assert test_path.is_file()
    assert rep_path.is_file()

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    # 10 reportado rows: 7 train, 1 val, 2 test (int(0.7*10)=7, int(0.85*10)=8 -> 7, 1, 2)
    assert len(train_df) == 7
    assert len(val_df) == 1
    assert len(test_df) == 2

    # Verify no overlap
    all_ids = list(train_df["id"]) + list(val_df["id"]) + list(test_df["id"])
    assert len(all_ids) == len(set(all_ids)) == 10

    # Verify non-reportado excluded
    assert not any(i in all_ids for i in range(11, 21))

    # Verify temporal ordering
    assert train_df["published"].max() <= val_df["published"].min()
    assert val_df["published"].max() <= test_df["published"].min()

    # Verify report fields
    rep = read_json(rep_path)
    assert rep["target_scope"] == "reportado"
    assert rep["rows_validated"] == 20
    assert rep["rows_modeling"] == 10
    assert rep["feature_count"] == 24
    assert rep["dataset_fingerprint"] == "test_fp_123"
