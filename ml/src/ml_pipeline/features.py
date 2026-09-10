from __future__ import annotations

import re
from typing import Final

import numpy as np
import pandas as pd

from ml_pipeline.data.collect import normalize_text

SKILLS: Final[dict[str, str]] = {
    "python": "python",
    "sql": "sql",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "spark": "spark",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "machine_learning": "machine learning",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "tableau": "tableau",
    "power_bi": "power bi",
}

CATEGORICAL_COLUMNS: Final[list[str]] = [
    "title",
    "country",
    "region",
    "experience_level",
    "work_mode",
    "company",
]

NUMERIC_COLUMNS: Final[list[str]] = [
    "company_is_agency",
    "experience_years",
    "experience_years_missing",
    "skill_python",
    "skill_sql",
    "skill_aws",
    "skill_azure",
    "skill_gcp",
    "skill_spark",
    "skill_docker",
    "skill_kubernetes",
    "skill_machine_learning",
    "skill_pytorch",
    "skill_tensorflow",
    "skill_tableau",
    "skill_power_bi",
    "published_year",
    "published_month",
]

FEATURE_COLUMNS: Final[list[str]] = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

RAW_REQUIRED_COLUMNS: Final[list[str]] = [
    "title",
    "countries",
    "regions",
    "experience_level",
    "has_remote",
    "work_mode",
    "company",
    "company_is_agency",
    "experience_years",
    "tags",
    "published",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the canonical 24 modeling features from raw vacancy attributes.

    This function is pure with respect to the input DataFrame, deterministic,
    and stateless. It accepts raw tabular records and produces the exact 24-feature contract.
    """
    out = pd.DataFrame(index=df.index)

    # 1. title: normalized text, empty -> 'desconocido'
    raw_title = df["title"] if "title" in df.columns else pd.Series("", index=df.index, dtype=object)
    out["title"] = normalize_text(raw_title).replace("", "desconocido")

    # 2. country: first element of countries separated by '|', empty/null -> 'desconocido', no normalize_text
    raw_countries = df["countries"] if "countries" in df.columns else pd.Series("desconocido", index=df.index, dtype=object)
    out["country"] = (
        raw_countries.fillna("desconocido")
        .astype(str)
        .str.split("|")
        .str[0]
        .str.strip()
        .replace("", "desconocido")
    )

    # 3. region: first element of regions separated by '|', empty/null -> 'desconocido', fails if column absent
    if "regions" not in df.columns:
        raise KeyError("Required column 'regions' is missing from DataFrame")
    out["region"] = (
        df["regions"]
        .fillna("desconocido")
        .astype(str)
        .str.split("|")
        .str[0]
        .str.strip()
        .replace("", "desconocido")
    )

    # 4. experience_level: fillna('desconocido').astype(str), preserves empty string
    raw_exp = df["experience_level"] if "experience_level" in df.columns else pd.Series("desconocido", index=df.index, dtype=object)
    out["experience_level"] = raw_exp.fillna("desconocido").astype(str)

    # 5. work_mode: derived from has_remote and work_mode numeric code
    raw_remote = df["has_remote"] if "has_remote" in df.columns else pd.Series(False, index=df.index)
    remote = pd.Series(np.where(raw_remote.isna(), False, raw_remote), index=df.index).astype(bool)

    raw_work = df["work_mode"] if "work_mode" in df.columns else pd.Series(np.nan, index=df.index)
    work = pd.to_numeric(raw_work, errors="coerce")

    out["work_mode"] = np.select(
        [~remote, work.eq(1), work.eq(2), work.eq(3)],
        ["presencial", "híbrido", "remoto", "remoto_global"],
        default="remoto_sin_detalle",
    )

    # 6. company: normalized text, empty -> 'desconocido'
    raw_company = df["company"] if "company" in df.columns else pd.Series("", index=df.index, dtype=object)
    out["company"] = normalize_text(raw_company).replace("", "desconocido")

    # 7. company_is_agency: fillna(False).astype(int)
    raw_agency = df["company_is_agency"] if "company_is_agency" in df.columns else pd.Series(False, index=df.index)
    out["company_is_agency"] = raw_agency.fillna(False).astype(int)

    # 8. experience_years: bounded between 0 and 50, otherwise NaN
    raw_years = df["experience_years"] if "experience_years" in df.columns else pd.Series(np.nan, index=df.index)
    years = pd.to_numeric(raw_years, errors="coerce")
    out["experience_years"] = years.where(years.between(0, 50))

    # 9. experience_years_missing: 1 if experience_years is NaN else 0
    out["experience_years_missing"] = out["experience_years"].isna().astype(int)

    # 10-22. skills: 13 binary features based on tag search
    raw_tags = df["tags"] if "tags" in df.columns else pd.Series("", index=df.index, dtype=object)
    tags = raw_tags.fillna("").astype(str).str.lower()
    for name, token in SKILLS.items():
        out[f"skill_{name}"] = tags.str.contains(re.escape(token), regex=True).astype(int)

    # 23-24. published_year and published_month (without replacing NaT with 0)
    pub = df["published"] if "published" in df.columns else pd.Series(pd.NaT, index=df.index)
    if not pd.api.types.is_datetime64_any_dtype(pub):
        pub = pd.to_datetime(pub, errors="coerce", utc=True)
    out["published_year"] = pub.dt.year
    out["published_month"] = pub.dt.month

    # Strictly enforce exact order and column count of FEATURE_COLUMNS
    return out[FEATURE_COLUMNS]
