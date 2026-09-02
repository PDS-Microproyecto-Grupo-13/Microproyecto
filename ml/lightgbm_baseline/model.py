from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd


def to_lightgbm_frame(features: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    """Cast categorical columns to pandas ``category`` dtype for LightGBM.

    Call this once on the *full* (pre-split) feature frame, then slice the
    result with train/validation/test masks. Fitting the category levels on
    the full frame keeps the encoding identical across splits.

    Unlike CatBoost (consumes raw strings) LightGBM's categorical support 
    relies on integer category codes.
    """
    frame = features.copy()
    for column in categorical_features:
        frame[column] = frame[column].astype("category")
    return frame


def train_model(
    model_params: dict[str, Any],
    train_features: pd.DataFrame,
    train_target: Any,
    validation_features: pd.DataFrame,
    validation_target: Any,
    categorical_features: list[str],
) -> lgb.LGBMRegressor:
    params = dict(model_params)
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
    eval_metric = params.get("metric", "mae")

    callbacks = []
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))

    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_features,
        train_target,
        eval_set=[(validation_features, validation_target)],
        eval_metric=eval_metric,
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )
    return model


def feature_importance_frame(
    minimum_model: lgb.LGBMRegressor,
    second_model: lgb.LGBMRegressor,
    feature_names: list[str],
    second_name: str,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Feature importance table"""
    minimum_importance = minimum_model.booster_.feature_importance(importance_type=importance_type)
    second_importance = second_model.booster_.feature_importance(importance_type=importance_type)
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_minimum": minimum_importance,
            f"importance_{second_name}": second_importance,
        }
    )
    frame["importance_mean"] = frame.iloc[:, 1:3].mean(axis=1)
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)
