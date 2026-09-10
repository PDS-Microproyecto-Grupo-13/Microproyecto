from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.data.collect import (
    PreflightError,
    compute_targets,
    create_data_manifest,
    deduplicate_by_id,
    deduplicate_by_signature,
    filter_valid_and_inliers,
    normalize_text,
    preflight_snapshots,
)


def test_normalize_text_standardizes_strings() -> None:
    series = pd.Series([
        "  Señor Developer / Python  ",
        "Machine-Learning Engineer!!",
        None,
        "C++ & C#",
    ])
    normalized = normalize_text(series)
    assert normalized.iloc[0] == "senor developer python"
    assert normalized.iloc[1] == "machine learning engineer"
    assert normalized.iloc[2] == ""
    assert normalized.iloc[3] == "c c"


def test_deduplicate_by_id_favors_priority_completeness_and_date() -> None:
    raw = pd.DataFrame([
        {
            "id": 101,
            "_priority": 0,
            "_complete": 5,
            "published": pd.Timestamp("2026-08-16T10:00:00Z"),
            "title": "Backend Dev v1",
        },
        {
            "id": 101,
            "_priority": 1,
            "_complete": 6,
            "published": pd.Timestamp("2026-08-20T10:00:00Z"),
            "title": "Backend Dev v2",
        },
    ])
    deduped = deduplicate_by_id(raw)
    assert len(deduped) == 1
    assert deduped.iloc[0]["title"] == "Backend Dev v2"


def test_deduplicate_by_signature_removes_repostings() -> None:
    by_id = pd.DataFrame([
        {
            "id": 201,
            "apply_url": "https://example.com/job/123#section",
            "company": "Tech Corp",
            "title": "Data Scientist",
            "location": "Remote",
            "company_is_agency": False,
            "salary_min": None,
            "salary_max": None,
            "salary_min_usd": None,
            "salary_max_usd": None,
            "_complete": 5,
            "published": pd.Timestamp("2026-08-16T00:00:00Z"),
        },
        {
            "id": 202,
            "apply_url": "https://example.com/job/123",
            "company": "Tech Corp",
            "title": "Data Scientist",
            "location": "Remote",
            "company_is_agency": False,
            "salary_min": 100000,
            "salary_max": 150000,
            "salary_min_usd": 100000,
            "salary_max_usd": 150000,
            "_complete": 9,
            "published": pd.Timestamp("2026-08-20T00:00:00Z"),
        },
    ])
    deduped = deduplicate_by_signature(by_id)
    assert len(deduped) == 1
    assert deduped.iloc[0]["id"] == 202
    assert bool(deduped.iloc[0]["company_is_agency"]) is False


def test_compute_targets_assigns_correct_sources() -> None:
    frame = pd.DataFrame([
        {
            "salary_min": 1000,
            "salary_max": 2000,
            "salary_min_usd": 1000,
            "salary_max_usd": 2000,
        },
        {
            "salary_min": 1000,
            "salary_max": None,
            "salary_min_usd": 1000,
            "salary_max_usd": 2500,
        },
        {
            "salary_min": None,
            "salary_max": None,
            "salary_min_usd": 1200,
            "salary_max_usd": 2200,
        },
    ])
    res = compute_targets(frame)
    assert res.loc[0, "target_source"] == "reportado"
    assert res.loc[1, "target_source"] == "híbrido"
    assert res.loc[2, "target_source"] == "estimado"
    assert res.loc[0, "y_min_usd"] == 1000.0
    assert res.loc[0, "y_max_usd"] == 2000.0


def test_filter_valid_and_inliers_filters_bad_targets_and_outliers() -> None:
    frame = pd.DataFrame([
        # Incomplete (NaN max)
        {"y_min_usd": 50000.0, "y_max_usd": np.nan},
        # Non-positive
        {"y_min_usd": -100.0, "y_max_usd": 50000.0},
        # Inverted
        {"y_min_usd": 90000.0, "y_max_usd": 60000.0},
        # Normal inliers
        {"y_min_usd": 50000.0, "y_max_usd": 80000.0},
        {"y_min_usd": 60000.0, "y_max_usd": 90000.0},
        {"y_min_usd": 70000.0, "y_max_usd": 100000.0},
        {"y_min_usd": 80000.0, "y_max_usd": 110000.0},
        {"y_min_usd": 90000.0, "y_max_usd": 120000.0},
        # Extreme outlier (e.g. 500 million)
        {"y_min_usd": 500000000.0, "y_max_usd": 800000000.0},
    ])
    interim, stats = filter_valid_and_inliers(frame)
    assert stats["y_incompleta"] == 1
    assert stats["y_no_positiva"] == 1
    assert stats["y_invertida"] == 1
    assert stats["extremos_retirados"] == 1
    assert len(interim) == 5


def test_preflight_fails_when_csv_dvc_missing_csv(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    (source_dir / "jobs_2026-08-16.csv").write_text("id\n1", encoding="utf-8")
    (source_dir / "jobs_2026-08-16.csv.dvc").write_text("outs: []", encoding="utf-8")
    # Unmaterialized snapshot
    (source_dir / "jobs_2026-09-09.csv.dvc").write_text("outs: []", encoding="utf-8")

    with pytest.raises(PreflightError) as exc_info:
        preflight_snapshots(source_dir)

    err = str(exc_info.value)
    assert "Missing 1 materialized snapshot(s)" in err
    assert "jobs_2026-09-09.csv" in err
    assert "dvc pull" in err


def test_preflight_fails_when_no_csv_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "empty_raw"
    source_dir.mkdir()
    with pytest.raises(PreflightError) as exc_info:
        preflight_snapshots(source_dir)
    assert "No materialized jobs_*.csv snapshot files found" in str(exc_info.value)


def test_data_manifest_is_deterministic(tmp_path: Path) -> None:
    f1 = tmp_path / "jobs_2026-08-16.csv"
    f2 = tmp_path / "jobs_2026-08-20.csv"
    f1.write_text("id,title\n1,dev\n", encoding="utf-8")
    f2.write_text("id,title\n2,lead\n", encoding="utf-8")

    manifest_a = create_data_manifest([f2, f1], {"jobs_2026-08-16.csv": 1, "jobs_2026-08-20.csv": 1})
    manifest_b = create_data_manifest([f1, f2], {"jobs_2026-08-16.csv": 1, "jobs_2026-08-20.csv": 1})

    assert manifest_a == manifest_b
    assert manifest_a["snapshot_count"] == 2
    assert manifest_a["snapshots"][0]["name"] == "jobs_2026-08-16.csv"
    assert manifest_a["snapshots"][1]["name"] == "jobs_2026-08-20.csv"
    assert len(manifest_a["dataset_fingerprint"]) == 64
