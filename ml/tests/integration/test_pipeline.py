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
        # Additional reported record to anchor distribution
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
    ])
    snap2.to_csv(raw_dir / "jobs_2026-08-20.csv", index=False)
    (raw_dir / "jobs_2026-08-20.csv.dvc").write_text("outs:\n- path: jobs_2026-08-20.csv\n", encoding="utf-8")

    (tmp_path / "params.yaml").write_text("data: {}\n", encoding="utf-8")
    return tmp_path


def test_reproducible_pipeline_collect_and_validate(synthetic_foorilla_env: Path) -> None:
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
