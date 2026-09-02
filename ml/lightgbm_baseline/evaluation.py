from __future__ import annotations

import numpy as np
import pandas as pd

from ml.catboost_baseline.evaluation import regression_metrics

# This module is additive to ml.catboost_baseline.evaluation, not a
# replacement for it. The metric itself is model-agnostic,  so the
# CatBoost baseline could import from here too if the team wants it later.


def seen_company_mask(
    company_series: pd.Series,
    train_mask: pd.Series,
    evaluation_mask: pd.Series,
) -> pd.Series:
    """Boolean mask, aligned to `evaluation_mask`'s True rows, marking whether
    that row's company also appears somewhere in the training split.

    `company_series` must share the same index as `train_mask`/`evaluation_mask`
    (e.g. the raw `frame["company"]` column, before `build_features` folds it
    into the categorical feature set).
    """
    normalized = company_series.fillna("Sin información").astype(str)
    known_companies = set(normalized.loc[train_mask])
    evaluation_companies = normalized.loc[evaluation_mask]
    return evaluation_companies.isin(known_companies)


def company_generalization_metrics(
    company_series: pd.Series,
    metadata: pd.DataFrame,
    train_mask: pd.Series,
    evaluation_mask: pd.Series,
    predicted_minimum: np.ndarray,
    predicted_maximum: np.ndarray,
    minimum_size: int = 30,
) -> list[dict[str, object]]:
    """Regression metrics split by whether the company was seen in training,
    computed on the same evaluation rows (and in the same row order) as
    `predicted_minimum`/`predicted_maximum`.

    `metadata` and `predicted_*` must already be restricted to the evaluation
    split and share its row order (i.e. exactly what `run.py` passes to
    `regression_metrics` for the test-set metrics). `company_series` and the
    masks are indexed on the full (pre-split) frame, matching how
    `metadata`/`features` are built in run.py.
    """
    seen = seen_company_mask(company_series, train_mask, evaluation_mask)
    # `seen` is indexed like metadata.loc[evaluation_mask]; align by position
    # so it lines up with predicted_minimum/predicted_maximum, which are
    # plain numpy arrays in the same row order as `metadata`.
    seen_positions = seen.to_numpy()

    actual_minimum = metadata["y_min_usd"].to_numpy()
    actual_maximum = metadata["y_max_usd"].to_numpy()
    predicted_minimum = np.asarray(predicted_minimum)
    predicted_maximum = np.asarray(predicted_maximum)

    results: list[dict[str, object]] = []
    for label, positions in (("seen", seen_positions), ("unseen", ~seen_positions)):
        if positions.sum() < minimum_size:
            continue
        metrics = regression_metrics(
            actual_minimum[positions],
            actual_maximum[positions],
            predicted_minimum[positions],
            predicted_maximum[positions],
        )
        results.append(
            {
                "segment": "company_seen_in_train",
                "value": label,
                "rows": int(positions.sum()),
                **metrics,
            }
        )

    return results


def company_generalization_dummy_metrics(
    company_series: pd.Series,
    metadata_all: pd.DataFrame,
    train_mask: pd.Series,
    evaluation_mask: pd.Series,
    minimum_size: int = 30,
) -> list[dict[str, object]]:
    """Median-of-train dummy baseline, computed separately for the seen and
    unseen company subsets of the evaluation split. Mirrors run.py's overall
    dummy_test baseline, but per segment, so "% improvement over dummy" can
    be reported for the unseen-company population specifically, instead of
    only in aggregate (which can hide a bad unseen-company story behind a
    good seen-company one).

    `metadata_all` should span both `train_mask` and `evaluation_mask` (i.e.
    the unfiltered metadata frame passed to build_features), since the
    median is computed from train rows and applied to evaluation rows.
    """
    train_median_minimum = float(metadata_all.loc[train_mask, "y_min_usd"].median())
    train_median_maximum = float(metadata_all.loc[train_mask, "y_max_usd"].median())

    seen = seen_company_mask(company_series, train_mask, evaluation_mask)
    evaluation_metadata = metadata_all.loc[evaluation_mask]
    seen_positions = seen.to_numpy()

    actual_minimum = evaluation_metadata["y_min_usd"].to_numpy()
    actual_maximum = evaluation_metadata["y_max_usd"].to_numpy()

    results: list[dict[str, object]] = []
    for label, positions in (("seen", seen_positions), ("unseen", ~seen_positions)):
        rows = int(positions.sum())
        if rows < minimum_size:
            continue
        dummy_minimum = np.full(rows, train_median_minimum)
        dummy_maximum = np.full(rows, train_median_maximum)
        metrics = regression_metrics(
            actual_minimum[positions], actual_maximum[positions], dummy_minimum, dummy_maximum
        )
        results.append(
            {
                "segment": "company_seen_in_train",
                "value": label,
                "rows": rows,
                **metrics,
            }
        )
    return results


def company_generalization_summary(
    company_series: pd.Series,
    train_mask: pd.Series,
    evaluation_mask: pd.Series,
) -> dict[str, object]:
    """Quick coverage check: how much of the evaluation split is even
    affected by this question. Cheap to log alongside the metrics above so
    a small unseen-company count doesn't get over-interpreted.
    """
    seen = seen_company_mask(company_series, train_mask, evaluation_mask)
    return {
        "evaluation_rows": int(len(seen)),
        "seen_company_rows": int(seen.sum()),
        "unseen_company_rows": int((~seen).sum()),
        "unseen_company_pct": float(round(100 * (~seen).mean(), 2)) if len(seen) else None,
    }
