from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def effective_model_params(config: dict[str, Any]) -> dict[str, Any]:
    algorithm = config.get("algorithm")
    if algorithm == "logistic_regression":
        algo_config = config.get("logistic_regression")
        if not isinstance(algo_config, dict):
            algo_config = {}
        return {
            "random_state": int(algo_config.get("random_state", config.get("random_state", 42))),
            "C": float(algo_config.get("C", config.get("C", 1.0))),
            "max_iter": int(algo_config.get("max_iter", config.get("max_iter", 1000))),
        }
    if algorithm == "random_forest":
        algo_config = config.get("random_forest")
        if not isinstance(algo_config, dict):
            algo_config = {}
        max_depth = algo_config.get("max_depth", config.get("max_depth", 10))
        return {
            "random_state": int(algo_config.get("random_state", config.get("random_state", 42))),
            "n_estimators": int(algo_config.get("n_estimators", config.get("n_estimators", 200))),
            "max_depth": int(max_depth) if max_depth is not None else None,
            "min_samples_leaf": int(algo_config.get("min_samples_leaf", config.get("min_samples_leaf", 1))),
        }
    raise ValueError(
        f"Unsupported algorithm: {algorithm!r}. "
        "Supported algorithms are 'logistic_regression' and 'random_forest'."
    )


def _build_logistic_regression(config: dict[str, Any]) -> Pipeline:
    params = effective_model_params(config)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=params["C"],
                    max_iter=params["max_iter"],
                    random_state=params["random_state"],
                ),
            ),
        ]
    )


def _build_random_forest(config: dict[str, Any]) -> Pipeline:
    params = effective_model_params(config)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=params["random_state"],
                ),
            ),
        ]
    )


def build_model(config: dict[str, Any]) -> Pipeline:
    algorithm = config.get("algorithm")
    if algorithm == "logistic_regression":
        return _build_logistic_regression(config)
    if algorithm == "random_forest":
        return _build_random_forest(config)
    raise ValueError(
        f"Unsupported algorithm: {algorithm!r}. "
        "Supported algorithms are 'logistic_regression' and 'random_forest'."
    )


build_reference_model = build_model

__all__ = ["build_model", "build_reference_model", "effective_model_params"]
