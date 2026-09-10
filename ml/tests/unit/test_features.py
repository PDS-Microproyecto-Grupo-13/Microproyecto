import numpy as np
import pandas as pd
import pytest

from ml_pipeline.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    SKILLS,
    prepare_features,
)


def test_feature_contract_dimensions_and_structure() -> None:
    assert len(FEATURE_COLUMNS) == 24
    assert len(CATEGORICAL_COLUMNS) == 6
    assert len(NUMERIC_COLUMNS) == 18
    assert CATEGORICAL_COLUMNS + NUMERIC_COLUMNS == FEATURE_COLUMNS
    assert not any("salary" in col.lower() for col in FEATURE_COLUMNS)
    assert "id" not in FEATURE_COLUMNS
    assert "y_min_usd" not in FEATURE_COLUMNS
    assert "y_max_usd" not in FEATURE_COLUMNS


def test_prepare_features_produces_exact_contract_and_order() -> None:
    df = pd.DataFrame([
        {
            "title": "Senior Python Developer",
            "countries": "Spain | Portugal",
            "regions": "EMEA | Southern Europe",
            "experience_level": "SE",
            "has_remote": True,
            "work_mode": 2,
            "company": "Acme Software",
            "company_is_agency": False,
            "experience_years": 5.0,
            "tags": "Python|Docker|AWS",
            "published": "2026-08-16T10:00:00Z",
        }
    ])
    features = prepare_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[0, "title"] == "senior python developer"
    assert features.loc[0, "country"] == "Spain"
    assert features.loc[0, "region"] == "EMEA"
    assert features.loc[0, "experience_level"] == "SE"
    assert features.loc[0, "work_mode"] == "remoto"
    assert features.loc[0, "company"] == "acme software"
    assert features.loc[0, "company_is_agency"] == 0
    assert features.loc[0, "experience_years"] == 5.0
    assert features.loc[0, "experience_years_missing"] == 0
    assert features.loc[0, "skill_python"] == 1
    assert features.loc[0, "skill_aws"] == 1
    assert features.loc[0, "skill_docker"] == 1
    assert features.loc[0, "skill_kubernetes"] == 0
    assert features.loc[0, "published_year"] == 2026
    assert features.loc[0, "published_month"] == 8


def test_work_mode_rules_mapping() -> None:
    df = pd.DataFrame([
        {"has_remote": False, "work_mode": 2, "regions": "Europe"},  # presencial (has_remote is False)
        {"has_remote": True, "work_mode": 1, "regions": "Europe"},   # híbrido
        {"has_remote": True, "work_mode": 2, "regions": "Europe"},   # remoto
        {"has_remote": True, "work_mode": 3, "regions": "Europe"},   # remoto_global
        {"has_remote": True, "work_mode": 99, "regions": "Europe"},  # remoto_sin_detalle
        {"has_remote": None, "work_mode": 2, "regions": "Europe"},   # presencial (has_remote null -> False)
    ])
    features = prepare_features(df)
    modes = features["work_mode"].tolist()
    assert modes == ["presencial", "híbrido", "remoto", "remoto_global", "remoto_sin_detalle", "presencial"]


def test_experience_years_bounds_and_missing_flag() -> None:
    df = pd.DataFrame([
        {"experience_years": 0, "regions": "Europe"},
        {"experience_years": 50, "regions": "Europe"},
        {"experience_years": -1, "regions": "Europe"},    # outside -> NaN
        {"experience_years": 51, "regions": "Europe"},    # outside -> NaN
        {"experience_years": "abc", "regions": "Europe"}, # non-numeric -> NaN
        {"experience_years": None, "regions": "Europe"},  # missing -> NaN
    ])
    features = prepare_features(df)
    assert features.loc[0, "experience_years"] == 0.0
    assert features.loc[0, "experience_years_missing"] == 0
    assert features.loc[1, "experience_years"] == 50.0
    assert features.loc[1, "experience_years_missing"] == 0
    assert np.isnan(features.loc[2, "experience_years"])
    assert features.loc[2, "experience_years_missing"] == 1
    assert np.isnan(features.loc[3, "experience_years"])
    assert features.loc[3, "experience_years_missing"] == 1
    assert np.isnan(features.loc[4, "experience_years"])
    assert features.loc[4, "experience_years_missing"] == 1
    assert np.isnan(features.loc[5, "experience_years"])
    assert features.loc[5, "experience_years_missing"] == 1


def test_missing_values_and_empty_strings_handled_gracefully() -> None:
    df = pd.DataFrame([
        {
            "title": "   ",
            "countries": None,
            "regions": "",
            "experience_level": None,
            "has_remote": None,
            "work_mode": None,
            "company": None,
            "company_is_agency": None,
            "experience_years": None,
            "tags": None,
            "published": None,
        }
    ])
    features = prepare_features(df)
    assert features.loc[0, "title"] == "desconocido"
    assert features.loc[0, "country"] == "desconocido"
    assert features.loc[0, "region"] == "desconocido"
    assert features.loc[0, "experience_level"] == "desconocido"
    assert features.loc[0, "company"] == "desconocido"
    assert features.loc[0, "work_mode"] == "presencial"
    assert features.loc[0, "company_is_agency"] == 0
    assert np.isnan(features.loc[0, "published_year"])
    assert np.isnan(features.loc[0, "published_month"])
    for skill in SKILLS:
        assert features.loc[0, f"skill_{skill}"] == 0


def test_parity_exact_rules() -> None:
    # 1. 13 skills, 24 features, 6 categorical, 18 numeric
    assert len(SKILLS) == 13
    assert len(FEATURE_COLUMNS) == 24
    assert len(CATEGORICAL_COLUMNS) == 6
    assert len(NUMERIC_COLUMNS) == 18

    # 2. Country & Region: keep accents/capitalization of 1st element, empty -> desconocido
    # 3. Experience level: empty string remains empty string
    # 4. Title & Company: normalized text
    # 5. Skills substring semantics and tokens
    df = pd.DataFrame([
        {
            "title": " Senior Software ENGINEER! ",
            "countries": "México|USA",
            "regions": "LATAM | South America",
            "experience_level": "",  # empty string should remain empty
            "has_remote": True,
            "work_mode": 2,
            "company": " ACME Corp. ",
            "company_is_agency": False,
            "experience_years": 4,
            "tags": "i am a pythonista and use power bi and machine learning",
            "published": pd.NaT,
        },
        {
            "title": "Data Scientist",
            "countries": "",  # empty -> desconocido
            "regions": "",    # empty -> desconocido
            "experience_level": None,  # None -> desconocido
            "has_remote": False,
            "work_mode": None,
            "company": "Tech",
            "company_is_agency": True,
            "experience_years": None,
            "tags": "experience with machine-learning and powerbi",  # hyphenated / no-space
            "published": "2026-08-20T10:00:00Z",
        },
    ])

    out = prepare_features(df)

    # Row 0 assertions
    assert out.loc[0, "country"] == "México"  # not 'mexico'
    assert out.loc[0, "region"] == "LATAM"    # not 'latam'
    assert out.loc[0, "experience_level"] == ""  # empty remains empty
    assert out.loc[0, "title"] == "senior software engineer"
    assert out.loc[0, "company"] == "acme corp"
    assert out.loc[0, "skill_python"] == 1  # 'pythonista' contains 'python'
    assert out.loc[0, "skill_machine_learning"] == 1
    assert out.loc[0, "skill_power_bi"] == 1
    assert np.isnan(out.loc[0, "published_year"])  # NaT -> NaN
    assert np.isnan(out.loc[0, "published_month"])

    # Row 1 assertions
    assert out.loc[1, "country"] == "desconocido"
    assert out.loc[1, "region"] == "desconocido"
    assert out.loc[1, "experience_level"] == "desconocido"
    assert out.loc[1, "skill_machine_learning"] == 0  # 'machine-learning' does not match 'machine learning'
    assert out.loc[1, "skill_power_bi"] == 0          # 'powerbi' does not match 'power bi'
    assert out.loc[1, "published_year"] == 2026
    assert out.loc[1, "published_month"] == 8


def test_missing_regions_column_raises_key_error() -> None:
    df = pd.DataFrame([{"title": "Dev", "countries": "Spain"}])
    with pytest.raises(KeyError, match="regions"):
        prepare_features(df)

