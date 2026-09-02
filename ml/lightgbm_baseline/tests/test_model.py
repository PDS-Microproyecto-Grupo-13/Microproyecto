from __future__ import annotations

import numpy as np
import pandas as pd

from ml.lightgbm_baseline.model import feature_importance_frame, to_lightgbm_frame, train_model


def test_to_lightgbm_frame_casts_only_listed_columns() -> None:
    features = pd.DataFrame(
        {
            "role_family": ["Data Engineer", "Data Scientist", "Data Engineer", "Data Scientist"],
            "experience_years": [3.0, 5.0, 1.0, 8.0],
        }
    )
    converted = to_lightgbm_frame(features, ["role_family"])
    assert str(converted["role_family"].dtype) == "category"
    assert converted["experience_years"].dtype == features["experience_years"].dtype


def test_to_lightgbm_frame_shares_categories_across_a_later_slice() -> None:
    features = pd.DataFrame({"role_family": ["Data Engineer", "Data Scientist", "MLOps"]})
    converted = to_lightgbm_frame(features, ["role_family"])
    train_slice = converted.iloc[:2]
    # Category levels are fixed on the full frame, so a slice that drops one
    # level still knows about it (and would encode an unseen test value the
    # same way LightGBM expects: as a missing code, not a new column).
    assert list(train_slice["role_family"].cat.categories) == ["Data Engineer", "Data Scientist", "MLOps"]


def test_train_model_respects_early_stopping_and_predicts() -> None:
    rng = np.random.default_rng(0)
    n = 200
    features = pd.DataFrame(
        {
            "role_family": rng.choice(["Data Engineer", "Data Scientist"], size=n),
            "experience_years": rng.uniform(0, 15, size=n),
        }
    )
    features = to_lightgbm_frame(features, ["role_family"])
    target = features["experience_years"] * 1000 + rng.normal(0, 50, size=n)

    train_features, validation_features = features.iloc[:150], features.iloc[150:]
    train_target, validation_target = target.iloc[:150], target.iloc[150:]

    params = {
        "objective": "mae",
        "metric": "mae",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "random_state": 42,
        "verbose": -1,
        "early_stopping_rounds": 10,
    }
    model = train_model(params, train_features, train_target, validation_features, validation_target, ["role_family"])
    predictions = model.predict(validation_features)
    assert predictions.shape == (50,)
    assert model.best_iteration_ is not None
    assert model.best_iteration_ <= 200


def test_feature_importance_frame_is_sorted_and_named() -> None:
    rng = np.random.default_rng(1)
    n = 100
    features = pd.DataFrame({"a": rng.uniform(size=n), "b": rng.uniform(size=n)})
    minimum_target = features["a"] * 10
    second_target = features["b"] * 5

    params = {"n_estimators": 30, "num_leaves": 7, "verbose": -1, "random_state": 42}
    minimum_model = train_model(params, features, minimum_target, features, minimum_target, [])
    second_model = train_model(params, features, second_target, features, second_target, [])

    importance = feature_importance_frame(minimum_model, second_model, ["a", "b"], "width")
    assert list(importance.columns) == ["feature", "importance_minimum", "importance_width", "importance_mean"]
    assert importance["importance_mean"].is_monotonic_decreasing
