from pathlib import Path

import pandas as pd
import pytest

from ml_pipeline.common.io import read_json
from ml_pipeline.data.collect import collect
from ml_pipeline.data.validate import validate
from ml_pipeline.settings import Settings


@pytest.fixture
def synthetic_foorilla_env(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw" / "foorilla"
    raw_dir.mkdir(parents=True)

    # Snapshot 1
    snap1 = pd.DataFrame([
        # 1. Valid reported
        {
            "id": 1,
            "apply_url": "https://company-a.com/job1",
            "company": "Company A",
            "title": "Backend Dev",
            "location": "Remote",
            "published": "2026-08-16T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 60000,
            "salary_max": 90000,
            "salary_min_usd": 60000,
            "salary_max_usd": 90000,
        },
        # 2. Incomplete record for id=2 (will be updated in snapshot 2)
        {
            "id": 2,
            "apply_url": "https://company-b.com/job2",
            "company": "Company B",
            "title": "Data Eng",
            "location": "Madrid",
            "published": "2026-08-16T11:00:00Z",
            "company_is_agency": False,
            "salary_min": 70000,
            "salary_max": None,
            "salary_min_usd": 70000,
            "salary_max_usd": 95000,
        },
        # 3. Agency posting (will be deduplicated by signature in snapshot 2)
        {
            "id": 3,
            "apply_url": "https://company-c.com/job3#ref123",
            "company": "Company C",
            "title": "ML Engineer",
            "location": "Berlin",
            "published": "2026-08-16T12:00:00Z",
            "company_is_agency": False,
            "salary_min": None,
            "salary_max": None,
            "salary_min_usd": 80000,
            "salary_max_usd": 110000,
        },
        # 4. Target negative
        {
            "id": 4,
            "apply_url": "https://company-d.com/job4",
            "company": "Company D",
            "title": "Frontend",
            "location": "Paris",
            "published": "2026-08-16T13:00:00Z",
            "company_is_agency": False,
            "salary_min": -5000,
            "salary_max": 50000,
            "salary_min_usd": -5000,
            "salary_max_usd": 50000,
        },
        # 5. Target inverted
        {
            "id": 5,
            "apply_url": "https://company-e.com/job5",
            "company": "Company E",
            "title": "DevOps",
            "location": "London",
            "published": "2026-08-16T14:00:00Z",
            "company_is_agency": False,
            "salary_min": 120000,
            "salary_max": 80000,
            "salary_min_usd": 120000,
            "salary_max_usd": 80000,
        },
        # 6. Extreme Outlier
        {
            "id": 6,
            "apply_url": "https://company-f.com/job6",
            "company": "Company F",
            "title": "Architect",
            "location": "Tokyo",
            "published": "2026-08-16T15:00:00Z",
            "company_is_agency": False,
            "salary_min": 500000000,
            "salary_max": 900000000,
            "salary_min_usd": 500000000,
            "salary_max_usd": 900000000,
        },
    ])
    snap1["regions"] = "Europe"
    snap1.to_csv(raw_dir / "jobs_2026-08-16.csv", index=False)
    (raw_dir / "jobs_2026-08-16.csv.dvc").write_text("outs:\n- path: jobs_2026-08-16.csv\n", encoding="utf-8")

    # Snapshot 2
    snap2 = pd.DataFrame([
        # id=2 updated with more complete data and later date
        {
            "id": 2,
            "apply_url": "https://company-b.com/job2",
            "company": "Company B",
            "title": "Data Eng",
            "location": "Madrid",
            "published": "2026-08-20T09:00:00Z",
            "company_is_agency": False,
            "salary_min": 70000,
            "salary_max": 100000,
            "salary_min_usd": 70000,
            "salary_max_usd": 100000,
        },
        # Direct employer reposting of id=3
        {
            "id": 7,
            "apply_url": "https://company-c.com/job3",
            "company": "Company C",
            "title": "ML Engineer",
            "location": "Berlin",
            "published": "2026-08-20T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 85000,
            "salary_max": 115000,
            "salary_min_usd": 85000,
            "salary_max_usd": 115000,
        },
        # Estimated salary
        {
            "id": 8,
            "apply_url": "https://company-g.com/job8",
            "company": "Company G",
            "title": "Security",
            "location": "Remote",
            "published": "2026-08-20T11:00:00Z",
            "company_is_agency": False,
            "salary_min": None,
            "salary_max": None,
            "salary_min_usd": 75000,
            "salary_max_usd": 105000,
        },
        # Hybrid salary
        {
            "id": 9,
            "apply_url": "https://company-h.com/job9",
            "company": "Company H",
            "title": "Product Manager",
            "location": "Remote",
            "published": "2026-08-20T12:00:00Z",
            "company_is_agency": False,
            "salary_min": 80000,
            "salary_max": None,
            "salary_min_usd": 80000,
            "salary_max_usd": 120000,
        },
        # Additional reported records to anchor distribution and test temporal splits
        {
            "id": 10,
            "apply_url": "https://company-j.com/job10",
            "company": "Company J",
            "title": "QA Lead",
            "location": "Remote",
            "published": "2026-08-20T13:00:00Z",
            "company_is_agency": False,
            "salary_min": 65000,
            "salary_max": 95000,
            "salary_min_usd": 65000,
            "salary_max_usd": 95000,
        },
        {
            "id": 11,
            "apply_url": "https://company-k.com/job11",
            "company": "Company K",
            "title": "Data Analyst",
            "location": "Remote",
            "published": "2026-08-21T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 55000,
            "salary_max": 85000,
            "salary_min_usd": 55000,
            "salary_max_usd": 85000,
        },
        {
            "id": 12,
            "apply_url": "https://company-l.com/job12",
            "company": "Company L",
            "title": "Cloud Architect",
            "location": "Madrid",
            "published": "2026-08-22T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 90000,
            "salary_max": 130000,
            "salary_min_usd": 90000,
            "salary_max_usd": 130000,
        },
        {
            "id": 13,
            "apply_url": "https://company-m.com/job13",
            "company": "Company M",
            "title": "Security Specialist",
            "location": "Berlin",
            "published": "2026-08-23T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 75000,
            "salary_max": 105000,
            "salary_min_usd": 75000,
            "salary_max_usd": 105000,
        },
        {
            "id": 14,
            "apply_url": "https://company-n.com/job14",
            "company": "Company N",
            "title": "Fullstack Dev",
            "location": "Paris",
            "published": "2026-08-24T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 68000,
            "salary_max": 98000,
            "salary_min_usd": 68000,
            "salary_max_usd": 98000,
        },
        {
            "id": 15,
            "apply_url": "https://company-o.com/job15",
            "company": "Company O",
            "title": "Site Reliability Eng",
            "location": "Remote",
            "published": "2026-08-25T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 82000,
            "salary_max": 112000,
            "salary_min_usd": 82000,
            "salary_max_usd": 112000,
        },
    ])
    extra_rows = [
        {
            "id": i,
            "apply_url": f"https://company-extra-{i}.com/job{i}",
            "company": f"Company {chr(65 + (i % 20))}",
            "title": "Engineer",
            "location": "Remote",
            "published": f"2026-08-{25 + ((i - 16) // 5):02d}T10:00:00Z",
            "company_is_agency": False,
            "salary_min": 60000 + (i * 500),
            "salary_max": 90000 + (i * 500),
            "salary_min_usd": 60000 + (i * 500),
            "salary_max_usd": 90000 + (i * 500),
        }
        for i in range(16, 46)
    ]
    snap2 = pd.concat([snap2, pd.DataFrame(extra_rows)], ignore_index=True)
    snap2["regions"] = "Europe"
    snap2.to_csv(raw_dir / "jobs_2026-08-20.csv", index=False)
    (raw_dir / "jobs_2026-08-20.csv.dvc").write_text("outs:\n- path: jobs_2026-08-20.csv\n", encoding="utf-8")

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
    n_estimators: 10
    num_leaves: 15
    min_child_samples: 2
    verbosity: -1

qualification:
  min_improvement_vs_baseline: -1.0
  max_temporal_gap: 10.0
  backtest_train_ratio: 0.80
  uncertainty_quantile: 0.80
""",
        encoding="utf-8",
    )
    return tmp_path


def test_reproducible_pipeline_collect_validate_and_preprocess(synthetic_foorilla_env: Path) -> None:
    from ml_pipeline.data.preprocess import preprocess
    from ml_pipeline.features import FEATURE_COLUMNS

    settings = Settings.load(synthetic_foorilla_env)

    # 1. Execute collect
    collect(settings)

    interim_path = synthetic_foorilla_env / "data/interim/foorilla_consolidated.parquet"
    manifest_path = synthetic_foorilla_env / "artifacts/reports/data_manifest.json"

    assert interim_path.is_file()
    assert manifest_path.is_file()

    manifest = read_json(manifest_path)
    assert manifest["source"] == "foorilla"
    assert manifest["snapshot_count"] == 2
    assert len(manifest["dataset_fingerprint"]) == 64

    interim_df = pd.read_parquet(interim_path)
    # Check deduplication by id: id=2 must have the updated data from snapshot 2 (y_max_usd == 100000)
    id2_row = interim_df[interim_df["id"] == 2]
    assert len(id2_row) == 1
    assert id2_row.iloc[0]["y_max_usd"] == 100000.0

    # Check deduplication by signature: id=3 should have been replaced by id=7 (non-agency, reported)
    assert 3 not in interim_df["id"].values
    assert 7 in interim_df["id"].values

    # Check targets hygiene: id=4 (negative), id=5 (inverted), id=6 (outlier) must not be present
    assert 4 not in interim_df["id"].values
    assert 5 not in interim_df["id"].values
    assert 6 not in interim_df["id"].values

    # Check that all target_source categories are preserved in interim dataset
    sources = set(interim_df["target_source"].unique())
    assert {"reportado", "híbrido", "estimado"}.issubset(sources)

    # 2. Execute validate
    validate(settings)

    validated_path = synthetic_foorilla_env / "data/validated/dataset.parquet"
    validation_rep_path = synthetic_foorilla_env / "artifacts/reports/validation.json"

    assert validated_path.is_file()
    assert validation_rep_path.is_file()

    val_report = read_json(validation_rep_path)
    assert val_report["valid"] is True
    assert val_report["rows"] == len(interim_df)
    assert "reportado" in val_report["target_source_counts"]
    assert "y_min_usd" in val_report["target_statistics"]

    # 3. Execute preprocess
    preprocess(settings)

    train_path = synthetic_foorilla_env / "data/processed/train.parquet"
    val_part_path = synthetic_foorilla_env / "data/processed/validation.parquet"
    test_part_path = synthetic_foorilla_env / "data/processed/test.parquet"
    limits_path = synthetic_foorilla_env / "artifacts/reports/train_limits.json"
    prep_rep_path = synthetic_foorilla_env / "artifacts/reports/preprocess.json"

    assert train_path.is_file()
    assert val_part_path.is_file()
    assert test_part_path.is_file()
    assert limits_path.is_file()
    assert prep_rep_path.is_file()

    train_df = pd.read_parquet(train_path)
    val_part_df = pd.read_parquet(val_part_path)
    test_part_df = pd.read_parquet(test_part_path)

    # Verify feature columns contract and order
    expected_cols = ["id", "published"] + FEATURE_COLUMNS + ["y_min_usd", "y_max_usd"]
    assert list(train_df.columns) == expected_cols
    assert list(val_part_df.columns) == expected_cols
    assert list(test_part_df.columns) == expected_cols

    # Verify no overlap across splits
    all_processed_ids = list(train_df["id"]) + list(val_part_df["id"]) + list(test_part_df["id"])
    assert len(all_processed_ids) == len(set(all_processed_ids))

    # Verify temporal order
    assert train_df["published"].max() <= val_part_df["published"].min()
    assert val_part_df["published"].max() <= test_part_df["published"].min()

    # Verify train limits
    limits = read_json(limits_path)
    assert limits["floor"] >= 1000.0
    assert limits["floor"] < limits["ceiling"]
    assert limits["source_split"] == "train"

    # Verify preprocess report
    prep_rep = read_json(prep_rep_path)
    assert prep_rep["target_scope"] == "reportado"
    assert prep_rep["feature_count"] == 24
    assert prep_rep["categorical_feature_count"] == 6
    assert prep_rep["numeric_feature_count"] == 18

    # 4. Execute qualify
    from ml_pipeline.modeling.qualify import qualify

    qualify_rep = qualify(settings)

    qual_path = synthetic_foorilla_env / "artifacts/reports/qualification.json"
    calib_path = synthetic_foorilla_env / "artifacts/reports/uncertainty_calibration.json"

    assert qual_path.is_file()
    assert calib_path.is_file()

    qual_data = read_json(qual_path)
    assert qual_data["algorithm"] == "lightgbm"
    assert qual_data["baseline_mae"] > 0.0
    assert qual_data["validation_mae"] > 0.0
    assert qual_data["uncertainty_margin"] > 0.0
    assert qual_data["nominal_coverage"] == 0.80
    assert isinstance(qual_data["eligible"], bool)

    calib_data = read_json(calib_path)
    assert calib_data["nominal_coverage"] == 0.80
    assert calib_data["quantile_method"] == "higher"
    assert calib_data["source_split"] == "validation"
    assert calib_data["validation_rows"] == len(val_part_df)

    # 5. Execute train
    from ml_pipeline.modeling.train import predict_range, train
    import joblib

    train_rep = train(settings)

    model_path = synthetic_foorilla_env / "artifacts/work/model/model.joblib"
    training_rep_path = synthetic_foorilla_env / "artifacts/reports/training.json"

    assert model_path.is_file()
    assert training_rep_path.is_file()

    training_meta = read_json(training_rep_path)
    assert training_meta["algorithm"] == "lightgbm"
    assert training_meta["train_rows"] == len(train_df)
    assert training_meta["validation_rows"] == len(val_part_df)
    assert training_meta["final_training_rows"] == len(train_df) + len(val_part_df)
    assert training_meta["qualification"]["eligible"] is True

    # Smoke test on reloaded model
    bundle = joblib.load(model_path)
    smoke_preds = predict_range(val_part_df.head(2), bundle)
    assert smoke_preds.shape == (2, 2)
    assert (smoke_preds > 0).all()
    assert (smoke_preds[:, 0] <= smoke_preds[:, 1]).all()


