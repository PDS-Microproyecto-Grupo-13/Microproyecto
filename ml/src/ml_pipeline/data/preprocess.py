from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml_pipeline.common.io import ensure_parent, read_json, write_json
from ml_pipeline.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    prepare_features,
)
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)

METADATA_COLUMNS: list[str] = ["id", "published"]
TARGET_COLUMNS: list[str] = ["y_min_usd", "y_max_usd"]


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split proportions."""
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0):
        raise ValueError(
            f"Split ratios must be strictly between 0 and 1. "
            f"Got train={train_ratio}, validation={val_ratio}, test={test_ratio}"
        )
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0. Got sum={total:.6f}")


def compute_train_limits(train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute operational salary floor and ceiling exclusively using train split."""
    if train_df.empty:
        raise ValueError("Cannot calculate train limits on empty training dataframe")

    floor = max(1000.0, float(train_df["y_min_usd"].quantile(0.001)))
    ceiling = float(train_df["y_max_usd"].quantile(0.999))

    if floor <= 0 or ceiling <= 0 or floor >= ceiling:
        raise ValueError(f"Invalid train limits computed: floor={floor}, ceiling={ceiling}")

    return {
        "floor": floor,
        "ceiling": ceiling,
        "source_split": "train",
        "floor_quantile": 0.001,
        "ceiling_quantile": 0.999,
    }


def preprocess(settings: Settings) -> None:
    source_path = settings.path("data/validated/dataset.parquet")
    train_out = settings.path("data/processed/train.parquet")
    val_out = settings.path("data/processed/validation.parquet")
    test_out = settings.path("data/processed/test.parquet")
    limits_out = settings.path("artifacts/reports/train_limits.json")
    report_out = settings.path("artifacts/reports/preprocess.json")
    manifest_path = settings.path("artifacts/reports/data_manifest.json")

    LOGGER.info("preprocess | input=%s", source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Validated dataset not found: {source_path}")

    data_config = settings.section("data")
    train_ratio = float(data_config.get("train_ratio", 0.70))
    val_ratio = float(data_config.get("validation_ratio", 0.15))
    test_ratio = float(data_config.get("test_ratio", 0.15))
    target_scope = str(data_config.get("target_scope", "reportado"))

    validate_ratios(train_ratio, val_ratio, test_ratio)

    validated_df = pd.read_parquet(source_path, engine="pyarrow")
    rows_validated = len(validated_df)

    # 1. Filter modeling universe
    reported_df = validated_df[validated_df["target_source"] == target_scope].copy()
    rows_modeling = len(reported_df)
    if reported_df.empty:
        raise ValueError(f"No records found with target_source == {target_scope!r}")

    # 2. Strict temporal sorting (preserve notebook stable sort order)
    null_published = int(reported_df["published"].isna().sum())
    duplicate_timestamps = int(len(reported_df) - reported_df["published"].nunique())
    sorted_df = reported_df.sort_values("published", na_position="first", kind="stable").copy()

    # 3. Temporal split: 70% train, 15% validation, 15% test
    n = len(sorted_df)
    a = int(train_ratio * n)
    b = int((train_ratio + val_ratio) * n)

    train_raw = sorted_df.iloc[:a].copy()
    val_raw = sorted_df.iloc[a:b].copy()
    test_raw = sorted_df.iloc[b:].copy()

    # Integrity assertions
    if train_raw.empty or val_raw.empty or test_raw.empty:
        raise ValueError("One or more splits are empty")
    if len(train_raw) + len(val_raw) + len(test_raw) != n:
        raise ValueError("Sum of split partitions does not match modeling rows count")

    train_ids = set(train_raw["id"])
    val_ids = set(val_raw["id"])
    test_ids = set(test_raw["id"])
    if not train_ids.isdisjoint(val_ids) or not train_ids.isdisjoint(test_ids) or not val_ids.isdisjoint(test_ids):
        raise ValueError("Data leakage: Overlapping records detected across splits")

    # 4. Compute operational limits using train only
    train_limits = compute_train_limits(train_raw)

    # 5. Extract 24 features for each split
    processed_parts: dict[str, pd.DataFrame] = {}
    for name, part in [("train", train_raw), ("validation", val_raw), ("test", test_raw)]:
        features = prepare_features(part)

        # Assert feature contract
        if len(features.columns) != 24:
            raise ValueError(f"Expected 24 features, got {len(features.columns)}")
        if list(features.columns) != FEATURE_COLUMNS:
            raise ValueError("Feature columns or order diverge from canonical contract")
        salary_cols = [c for c in features.columns if "salary" in c.lower()]
        if salary_cols:
            raise ValueError(f"Salary leakage in features: {salary_cols}")
        for target in TARGET_COLUMNS:
            if target in features.columns:
                raise ValueError(f"Target column '{target}' present in feature set")

        processed = pd.concat(
            [
                part[METADATA_COLUMNS].reset_index(drop=True),
                features.reset_index(drop=True),
                part[TARGET_COLUMNS].reset_index(drop=True),
            ],
            axis=1,
        )
        processed_parts[name] = processed

    # 6. Persist processed Parquet partitions
    ensure_parent(train_out)
    processed_parts["train"].to_parquet(train_out, index=False, engine="pyarrow")
    processed_parts["validation"].to_parquet(val_out, index=False, engine="pyarrow")
    processed_parts["test"].to_parquet(test_out, index=False, engine="pyarrow")
    LOGGER.info(
        "preprocess | saved_splits | train=%d val=%d test=%d",
        len(processed_parts["train"]),
        len(processed_parts["validation"]),
        len(processed_parts["test"]),
    )

    # 7. Persist train_limits.json
    write_json(limits_out, train_limits)
    LOGGER.info("preprocess | saved_limits | floor=%.2f ceiling=%.2f", train_limits["floor"], train_limits["ceiling"])

    # 8. Build and persist preprocess.json
    dataset_fingerprint = ""
    if manifest_path.is_file():
        manifest_data = read_json(manifest_path)
        dataset_fingerprint = manifest_data.get("dataset_fingerprint", "")

    preprocess_report = {
        "target_scope": target_scope,
        "rows_validated": rows_validated,
        "rows_modeling": rows_modeling,
        "split": {
            "train": {
                "rows": len(train_raw),
                "published_min": str(train_raw["published"].min()),
                "published_max": str(train_raw["published"].max()),
            },
            "validation": {
                "rows": len(val_raw),
                "published_min": str(val_raw["published"].min()),
                "published_max": str(val_raw["published"].max()),
            },
            "test": {
                "rows": len(test_raw),
                "published_min": str(test_raw["published"].min()),
                "published_max": str(test_raw["published"].max()),
            },
        },
        "published_nulls": null_published,
        "feature_count": len(FEATURE_COLUMNS),
        "categorical_feature_count": len(CATEGORICAL_COLUMNS),
        "numeric_feature_count": len(NUMERIC_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "train_limits": {
            "floor": train_limits["floor"],
            "ceiling": train_limits["ceiling"],
        },
        "dataset_fingerprint": dataset_fingerprint,
    }
    write_json(report_out, preprocess_report)
    LOGGER.info("preprocess | saved_report | path=%s", report_out)
