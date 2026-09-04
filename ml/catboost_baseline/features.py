from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import normalize_text, parse_boolean

TOOL_ALIASES = {
    "Python": ["python"],
    "SQL": ["sql"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Spark": ["spark", "apache spark", "pyspark"],
    "Databricks": ["databricks"],
    "Snowflake": ["snowflake"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes"],
    "Airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt"],
    "Kafka": ["kafka", "apache kafka"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Java": ["java"],
    "R": ["r"],
    "CI/CD": ["ci/cd"],
}

KNOWLEDGE_ALIASES = {
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "NLP": ["natural language processing"],
    "LLM": ["large language models", "llm"],
    "IA generativa": ["generative ai", "genai"],
    "Computer Vision": ["computer vision"],
    "Estadística": ["statistics", "statistical analysis"],
    "Series de tiempo": ["time series", "time series analysis"],
    "ETL/ELT": ["etl", "elt"],
    "Modelado de datos": ["data modeling"],
    "MLOps": ["mlops"],
    "Business Intelligence": ["business intelligence"],
    "Big Data": ["big data"],
}

TOPIC_AREAS = [
    "Data Engineering",
    "Big Data",
    "Finance & Fintech",
    "MLOps",
    "Software Engineering & Development",
    "Data Science",
    "Artificial Intelligence",
    "Machine Learning",
]

BASE_CATEGORICAL_FEATURES = [
    "experience_level",
    "work_modality",
    "primary_country",
    "role_family",
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def role_family(value: str) -> str:
    if "mlops" in value:
        return "MLOps"
    if "machine learning" in value and ("engineer" in value or "developer" in value):
        return "ML Engineer"
    if re.search(r"\b(ai|artificial intelligence|genai|generative ai)\b", value) and (
        "engineer" in value or "developer" in value
    ):
        return "AI Engineer"
    if "data engineer" in value or "analytics engineer" in value:
        return "Data Engineer"
    if "data scientist" in value:
        return "Data Scientist"
    if (
        "data analyst" in value
        or "business intelligence analyst" in value
        or "bi analyst" in value
    ):
        return "Data/BI Analyst"
    if "data architect" in value:
        return "Data Architect"
    if "software engineer" in value or "software developer" in value:
        return "Software Engineer"
    return "Otra familia"


def build_features(
    frame: pd.DataFrame, include_company: bool = True
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Create a compact tabular feature set aligned with the current EDA."""

    features, categorical = build_inference_features(frame, include_company)
    published = pd.to_datetime(frame["published"], errors="coerce", utc=True)
    metadata = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "published": published,
            "target_source": frame["target_source"].astype(str),
            "experience_level": features["experience_level"],
            "primary_country": features["primary_country"],
            "role_family": features["role_family"],
            "y_min_usd": frame["y_min_usd"].astype(float),
            "y_max_usd": frame["y_max_usd"].astype(float),
        },
        index=frame.index,
    )
    return features, categorical, metadata


def build_inference_features(
    frame: pd.DataFrame, include_company: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Build the same feature matrix from the public inference contract.

    The serving payload intentionally contains no identifiers or salary targets.
    Keeping this transformation beside the training transformation prevents drift
    when new model versions are packaged.
    """

    index = frame.index
    experience = pd.to_numeric(frame.get("experience_years"), errors="coerce")
    experience = experience.where(experience.between(0, 50))
    has_remote = parse_boolean(frame.get("has_remote", pd.Series(index=index))).fillna(False)
    work_mode = pd.to_numeric(frame.get("work_mode"), errors="coerce")
    modality = pd.Series(
        np.select(
            [
                ~has_remote.astype(bool),
                has_remote.astype(bool) & work_mode.eq(1),
                has_remote.astype(bool) & work_mode.eq(3),
                has_remote.astype(bool),
            ],
            ["Presencial", "Híbrido", "Remoto en cualquier lugar", "Remoto"],
            default="Sin información",
        ),
        index=index,
        dtype="string",
    )

    country_tokens = frame.get("countries", pd.Series(index=index, dtype="object")).fillna("").map(
        lambda value: [token.strip() for token in str(value).split("|") if token.strip()]
    )
    primary_country = country_tokens.map(
        lambda values: values[0] if values else "Sin información"
    )
    title_normalized = normalize_text(frame.get("title", pd.Series(index=index)))
    published = pd.to_datetime(frame["published"], errors="coerce", utc=True)

    features = pd.DataFrame(
        {
            "experience_level": frame.get(
                "experience_level", pd.Series(index=index, dtype="object")
            ).fillna("Sin información"),
            "experience_years": experience,
            "experience_years_missing": experience.isna().astype(int),
            "work_modality": modality,
            "primary_country": primary_country,
            "country_count": country_tokens.map(len).astype(int),
            "ambiguous_country": country_tokens.map(len).gt(1).astype(int),
            "role_family": title_normalized.map(role_family),
            "company_is_agency": parse_boolean(
                frame.get("company_is_agency", pd.Series(index=index))
            )
            .fillna(False)
            .astype(int),
            "publication_year": published.dt.year.astype("float64"),
            "publication_month": published.dt.month.astype("float64"),
        },
        index=index,
    )

    categorical = list(BASE_CATEGORICAL_FEATURES)
    if include_company:
        features["company"] = frame.get(
            "company", pd.Series(index=index, dtype="object")
        ).fillna("Sin información")
        categorical.append("company")

    tag_lookup: dict[str, str] = {}
    for prefix, mapping in (("tool", TOOL_ALIASES), ("knowledge", KNOWLEDGE_ALIASES)):
        for label, aliases in mapping.items():
            column = f"{prefix}_{slug(label)}"
            for alias in aliases:
                tag_lookup[alias.casefold()] = column

    tags = frame.get("tags", pd.Series(index=index, dtype="object")).fillna("").map(
        lambda value: {
            tag_lookup[token.strip().casefold()]
            for token in str(value).split("|")
            if token.strip().casefold() in tag_lookup
        }
    )
    for column in sorted(set(tag_lookup.values())):
        features[column] = tags.map(lambda values, name=column: int(name in values))

    topic_sets = frame.get("topics", pd.Series(index=index, dtype="object")).fillna("").map(
        lambda value: {token.strip() for token in str(value).split("|") if token.strip()}
    )
    for area in TOPIC_AREAS:
        column = f"topic_{slug(area)}"
        features[column] = topic_sets.map(lambda values, name=area: int(name in values))

    for column in categorical:
        features[column] = features[column].fillna("Sin información").astype(str)

    return features, categorical
