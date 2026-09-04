from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .evaluation import reconstruct_targets
from .features import build_inference_features

SERVING_COLUMNS = [
    "title",
    "experience_level",
    "experience_years",
    "has_remote",
    "work_mode",
    "countries",
    "company_is_agency",
    "company",
    "tags",
    "topics",
    "published",
]


def serving_example(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a target-free input example matching the public API contract."""

    example = frame.copy()
    defaults: dict[str, Any] = {
        "title": "Data Scientist",
        "experience_level": "SE",
        "experience_years": 5.0,
        "has_remote": True,
        "work_mode": None,
        "countries": "United States",
        "company_is_agency": False,
        "company": "Sin información",
        "tags": "python|sql",
        "topics": "Data Science|Machine Learning",
        "published": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    for column, default in defaults.items():
        if column not in example:
            example[column] = default
    example["experience_years"] = pd.to_numeric(
        example["experience_years"], errors="coerce"
    ).astype("float64")
    example["work_mode"] = pd.to_numeric(example["work_mode"], errors="coerce").astype(
        "float64"
    )
    example["has_remote"] = example["has_remote"].fillna(False).astype(bool)
    example["company_is_agency"] = example["company_is_agency"].fillna(False).astype(bool)
    for column in ("title", "experience_level", "countries", "company", "tags", "topics"):
        example[column] = example[column].fillna(str(defaults[column])).astype(str)
    published = pd.to_datetime(example["published"], errors="coerce", utc=True)
    example["published"] = published.map(
        lambda value: value.isoformat() if not pd.isna(value) else defaults["published"]
    )
    return example[SERVING_COLUMNS]


class SalaryRangePyFunc(mlflow.pyfunc.PythonModel):
    """Portable MLflow model containing preprocessing and both CatBoost heads."""

    def __init__(self, target_strategy: str, include_company: bool) -> None:
        self.target_strategy = target_strategy
        self.include_company = include_company
        self.minimum_model: CatBoostRegressor | None = None
        self.second_model: CatBoostRegressor | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        minimum = CatBoostRegressor()
        minimum.load_model(context.artifacts["minimum_model"])
        second = CatBoostRegressor()
        second.load_model(context.artifacts["second_model"])
        self.minimum_model = minimum
        self.second_model = second

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params
        if self.minimum_model is None or self.second_model is None:
            raise RuntimeError("Model artifacts have not been loaded")
        raw = serving_example(pd.DataFrame(model_input))
        features, _ = build_inference_features(raw, include_company=self.include_company)
        first = np.asarray(self.minimum_model.predict(features), dtype=float)
        second = np.asarray(self.second_model.predict(features), dtype=float)
        minimum, maximum, _, _ = reconstruct_targets(first, second, self.target_strategy)
        return pd.DataFrame(
            {
                "salary_min_usd": minimum,
                "salary_max_usd": maximum,
                "salary_midpoint_usd": (minimum + maximum) / 2,
            },
            index=model_input.index,
        )
