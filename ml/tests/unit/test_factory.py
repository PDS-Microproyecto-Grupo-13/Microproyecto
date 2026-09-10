from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline.modeling.factory import build_model, effective_model_params


def test_effective_model_params_logistic_regression() -> None:
    config = {
        "algorithm": "logistic_regression",
        "random_state": 42,
        "logistic_regression": {"C": 2.5, "max_iter": 300},
    }
    params = effective_model_params(config)
    assert params == {"random_state": 42, "C": 2.5, "max_iter": 300}


def test_effective_model_params_random_forest() -> None:
    config = {
        "algorithm": "random_forest",
        "random_state": 7,
        "random_forest": {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 3},
    }
    params = effective_model_params(config)
    assert params == {"random_state": 7, "n_estimators": 100, "max_depth": 8, "min_samples_leaf": 3}


def test_build_model_logistic_regression() -> None:
    config = {
        "algorithm": "logistic_regression",
        "random_state": 42,
        "logistic_regression": {
            "C": 0.5,
            "max_iter": 500,
        },
    }
    model = build_model(config)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["imputer", "scaler", "classifier"]
    assert isinstance(model.named_steps["imputer"], SimpleImputer)
    assert model.named_steps["imputer"].strategy == "median"
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    classifier = model.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == 0.5
    assert classifier.max_iter == 500
    assert classifier.random_state == 42


def test_build_model_random_forest() -> None:
    config = {
        "algorithm": "random_forest",
        "random_state": 42,
        "random_forest": {
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_leaf": 2,
        },
    }
    model = build_model(config)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["imputer", "classifier"]
    assert isinstance(model.named_steps["imputer"], SimpleImputer)
    assert model.named_steps["imputer"].strategy == "median"
    classifier = model.named_steps["classifier"]
    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 5
    assert classifier.min_samples_leaf == 2
    assert classifier.random_state == 42


@pytest.mark.parametrize(
    "bad_config, expected_match",
    [
        ({"algorithm": "unsupported"}, "Unsupported algorithm: 'unsupported'"),
        ({"algorithm": None}, "Unsupported algorithm: None"),
        ({}, "Unsupported algorithm: None"),
    ],
)
def test_build_model_unsupported_algorithm(bad_config: dict, expected_match: str) -> None:
    with pytest.raises(ValueError, match=expected_match):
        build_model(bad_config)


@pytest.mark.parametrize("algorithm", ["logistic_regression", "random_forest"])
def test_models_fit_predict_binary_with_encapsulated_preprocessing(algorithm: str) -> None:
    config = {
        "algorithm": algorithm,
        "random_state": 42,
        "logistic_regression": {"C": 1.0, "max_iter": 200},
        "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
    }
    model = build_model(config)

    # Data containing NaN to verify encapsulated SimpleImputer preprocessing
    X_train = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 20.0, np.nan, 40.0, 50.0, 60.0],
        }
    )
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    X_test = pd.DataFrame(
        {
            "f1": [np.nan, 5.5],
            "f2": [15.0, np.nan],
        }
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})
