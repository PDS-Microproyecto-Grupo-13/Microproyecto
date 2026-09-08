from __future__ import annotations

import numpy as np
import pandas as pd

from ml.catboost_baseline.data import prepare_model_data, temporal_split
from ml.catboost_baseline.evaluation import reconstruct_targets, regression_metrics
from ml.catboost_baseline.features import build_features


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "apply_url": "https://jobs.example/1",
                "company": "Example",
                "company_is_agency": "False",
                "title": "Senior Data Engineer",
                "location": "Bogota",
                "published": "2026-01-01",
                "salary_min": 100,
                "salary_max": 150,
                "salary_min_usd": 100,
                "salary_max_usd": 150,
                "experience_level": "SE",
                "experience_years": 6,
                "has_remote": True,
                "work_mode": 2,
                "countries": "Colombia",
                "topics": "Data Engineering|Machine Learning",
                "tags": "python|sql|docker",
                "_source_priority": 0,
                "_source_file": "old.csv",
            },
            {
                "id": 1,
                "apply_url": "https://jobs.example/1",
                "company": "Example",
                "company_is_agency": False,
                "title": "Senior Data Engineer",
                "location": "Bogota",
                "published": "2026-01-02",
                "salary_min": 110,
                "salary_max": 160,
                "salary_min_usd": 110,
                "salary_max_usd": 160,
                "experience_level": "SE",
                "experience_years": 6,
                "has_remote": True,
                "work_mode": 2,
                "countries": "Colombia",
                "topics": "Data Engineering|Machine Learning",
                "tags": "python|sql|docker",
                "_source_priority": 1,
                "_source_file": "new.csv",
            },
            {
                "id": 2,
                "apply_url": "https://jobs.example/2",
                "company": "Another",
                "company_is_agency": False,
                "title": "Data Scientist",
                "location": "Mexico City",
                "published": "2026-02-01",
                "salary_min": 90,
                "salary_max": 140,
                "salary_min_usd": 90,
                "salary_max_usd": 140,
                "experience_level": "MI",
                "experience_years": 4,
                "has_remote": False,
                "work_mode": np.nan,
                "countries": "Mexico",
                "topics": "Data Science",
                "tags": "python|machine learning",
                "_source_priority": 1,
                "_source_file": "new.csv",
            },
            {
                "id": 3,
                "apply_url": "https://jobs.example/3",
                "company": "Broken",
                "company_is_agency": False,
                "title": "Analyst",
                "location": "Lima",
                "published": "2026-03-01",
                "salary_min": 200,
                "salary_max": 100,
                "salary_min_usd": 200,
                "salary_max_usd": 100,
                "experience_level": "EN",
                "experience_years": 1,
                "has_remote": False,
                "work_mode": np.nan,
                "countries": "Peru",
                "topics": "Data Science",
                "tags": "sql",
                "_source_priority": 1,
                "_source_file": "new.csv",
            },
        ]
    )


def test_preparation_keeps_latest_id_and_valid_ordered_ranges() -> None:
    prepared = prepare_model_data(source_frame())
    assert prepared.audit["duplicate_ids_removed"] == 1
    assert prepared.audit["unordered_targets"] == 1
    assert prepared.frame["id"].tolist() == [1, 2]
    assert prepared.frame.set_index("id").loc[1, "y_min_usd"] == 110
    assert set(prepared.frame["target_source"]) == {"Reportado"}


def test_feature_builder_creates_catboost_categories_and_skill_flags() -> None:
    prepared = prepare_model_data(source_frame()).frame
    features, categorical, metadata = build_features(prepared)
    assert "company" in categorical
    assert "role_family" in categorical
    assert features.loc[prepared["id"].eq(1), "tool_python"].iloc[0] == 1
    assert features.loc[prepared["id"].eq(1), "role_family"].iloc[0] == "Data Engineer"
    assert metadata["y_max_usd"].gt(metadata["y_min_usd"]).all()


def test_minimum_width_reconstruction_is_always_ordered() -> None:
    predicted_minimum, predicted_maximum, _, _ = reconstruct_targets(
        np.log1p([100, 200]), np.log1p([50, 0]), "minimum_width_log"
    )
    assert np.allclose(predicted_minimum, [100, 200])
    assert np.allclose(predicted_maximum, [150, 200])
    assert np.all(predicted_maximum >= predicted_minimum)


def test_metrics_compare_range_and_detect_crossing() -> None:
    metrics = regression_metrics(
        np.array([100.0, 200.0]),
        np.array([150.0, 260.0]),
        np.array([110.0, 190.0]),
        np.array([160.0, 250.0]),
        raw_minimum=np.array([170.0, 270.0]),
        raw_maximum=np.array([160.0, 250.0]),
    )
    assert metrics["mae_y1_usd"] == 10
    assert metrics["mae_y2_usd"] == 10
    assert metrics["raw_crossing_rate"] == 1


def test_temporal_split_uses_fixed_cutoffs() -> None:
    frame = pd.DataFrame(
        {"published": pd.date_range("2026-01-01", periods=20, tz="UTC", freq="D")}
    )
    split, cutoffs = temporal_split(frame, 0.70, 0.15)
    assert split.value_counts().to_dict() == {"train": 14, "validation": 3, "test": 3}
    assert cutoffs["train_cutoff"].startswith("2026-01-14")
