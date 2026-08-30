from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def reconstruct_targets(
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(first_prediction, dtype=float)
    second = np.asarray(second_prediction, dtype=float)
    if strategy == "minimum_width_log":
        predicted_minimum = np.expm1(np.clip(first, 0, None))
        predicted_width = np.expm1(np.clip(second, 0, None))
        predicted_maximum = predicted_minimum + predicted_width
        return predicted_minimum, predicted_maximum, predicted_minimum, predicted_maximum
    if strategy == "direct":
        raw_minimum = np.clip(first, 0, None)
        raw_maximum = np.clip(second, 0, None)
        corrected_minimum = np.minimum(raw_minimum, raw_maximum)
        corrected_maximum = np.maximum(raw_minimum, raw_maximum)
        return corrected_minimum, corrected_maximum, raw_minimum, raw_maximum
    raise ValueError("features.target_strategy must be 'direct' or 'minimum_width_log'")


def training_targets(
    y_minimum: pd.Series, y_maximum: pd.Series, strategy: str
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    if strategy == "minimum_width_log":
        return (
            np.log1p(y_minimum.to_numpy(dtype=float)),
            np.log1p((y_maximum - y_minimum).to_numpy(dtype=float)),
            ("log_minimum", "log_width"),
        )
    if strategy == "direct":
        return (
            y_minimum.to_numpy(dtype=float),
            y_maximum.to_numpy(dtype=float),
            ("minimum", "maximum"),
        )
    raise ValueError("features.target_strategy must be 'direct' or 'minimum_width_log'")


def regression_metrics(
    actual_minimum: np.ndarray,
    actual_maximum: np.ndarray,
    predicted_minimum: np.ndarray,
    predicted_maximum: np.ndarray,
    raw_minimum: np.ndarray | None = None,
    raw_maximum: np.ndarray | None = None,
) -> dict[str, float]:
    actual_minimum = np.asarray(actual_minimum, dtype=float)
    actual_maximum = np.asarray(actual_maximum, dtype=float)
    predicted_minimum = np.asarray(predicted_minimum, dtype=float)
    predicted_maximum = np.asarray(predicted_maximum, dtype=float)
    raw_minimum = predicted_minimum if raw_minimum is None else np.asarray(raw_minimum)
    raw_maximum = predicted_maximum if raw_maximum is None else np.asarray(raw_maximum)

    actual_width = actual_maximum - actual_minimum
    predicted_width = predicted_maximum - predicted_minimum
    actual_midpoint = (actual_minimum + actual_maximum) / 2
    intersection = np.maximum(
        0, np.minimum(actual_maximum, predicted_maximum) - np.maximum(actual_minimum, predicted_minimum)
    )
    union = np.maximum(actual_maximum, predicted_maximum) - np.minimum(
        actual_minimum, predicted_minimum
    )
    interval_iou = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)

    mae_minimum = mean_absolute_error(actual_minimum, predicted_minimum)
    mae_maximum = mean_absolute_error(actual_maximum, predicted_maximum)
    return {
        "mae_y1_usd": float(mae_minimum),
        "mae_y2_usd": float(mae_maximum),
        "mae_mean_usd": float((mae_minimum + mae_maximum) / 2),
        "rmse_y1_usd": float(root_mean_squared_error(actual_minimum, predicted_minimum)),
        "rmse_y2_usd": float(root_mean_squared_error(actual_maximum, predicted_maximum)),
        "r2_y1": float(r2_score(actual_minimum, predicted_minimum)),
        "r2_y2": float(r2_score(actual_maximum, predicted_maximum)),
        "mae_width_usd": float(mean_absolute_error(actual_width, predicted_width)),
        "raw_crossing_rate": float(np.mean(raw_minimum > raw_maximum)),
        "midpoint_coverage": float(
            np.mean(
                (predicted_minimum <= actual_midpoint)
                & (actual_midpoint <= predicted_maximum)
            )
        ),
        "mean_interval_iou": float(np.mean(interval_iou)),
    }


def segment_metrics(
    metadata: pd.DataFrame,
    predicted_minimum: np.ndarray,
    predicted_maximum: np.ndarray,
    minimum_size: int,
) -> list[dict[str, object]]:
    working = metadata.copy()
    working["predicted_minimum"] = predicted_minimum
    working["predicted_maximum"] = predicted_maximum
    results: list[dict[str, object]] = []
    for column in ("target_source", "experience_level", "primary_country", "role_family"):
        for value, group in working.groupby(column, dropna=False):
            if len(group) < minimum_size:
                continue
            metrics = regression_metrics(
                group["y_min_usd"].to_numpy(),
                group["y_max_usd"].to_numpy(),
                group["predicted_minimum"].to_numpy(),
                group["predicted_maximum"].to_numpy(),
            )
            results.append(
                {
                    "segment": column,
                    "value": str(value),
                    "rows": int(len(group)),
                    "mae_y1_usd": metrics["mae_y1_usd"],
                    "mae_y2_usd": metrics["mae_y2_usd"],
                    "mae_mean_usd": metrics["mae_mean_usd"],
                    "mae_width_usd": metrics["mae_width_usd"],
                }
            )
    return results
