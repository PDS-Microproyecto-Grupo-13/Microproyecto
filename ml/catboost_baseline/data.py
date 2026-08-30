from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparationResult:
    frame: pd.DataFrame
    audit: dict[str, object]


def discover_csv_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV files matched {pattern!r} under {input_dir}. "
            "Run dvc pull for the Foorilla CSV pointers first."
        )
    return files


def load_raw_data(files: list[Path], max_rows_per_file: int | None = None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for priority, path in enumerate(files):
        part = pd.read_csv(path, low_memory=False, nrows=max_rows_per_file)
        part["_source_file"] = path.name
        part["_source_priority"] = priority
        parts.append(part)
    if not parts:
        raise ValueError("At least one source file is required")
    return pd.concat(parts, ignore_index=True)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def parse_boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    return normalized.map(mapping).astype("boolean")


def _required_columns(frame: pd.DataFrame) -> None:
    required = {
        "id",
        "apply_url",
        "company",
        "company_is_agency",
        "title",
        "location",
        "published",
        "salary_min",
        "salary_max",
        "salary_min_usd",
        "salary_max_usd",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")


def prepare_model_data(raw: pd.DataFrame, outlier_iqr_multiplier: float = 3.0) -> PreparationResult:
    """Integrate cuts, deduplicate vacancies, build targets, and remove invalid ranges.

    The operations intentionally mirror the EDA decisions in notebooks/02_eda.ipynb,
    with an explicit boolean parser to avoid treating the string ``"False"`` as true.
    """

    _required_columns(raw)
    frame = raw.copy()
    source_rows = len(frame)
    frame["published"] = pd.to_datetime(frame["published"], errors="coerce", utc=True)
    source_columns = [name for name in frame.columns if not name.startswith("_")]
    frame["_completeness"] = frame[source_columns].notna().sum(axis=1)

    by_id = (
        frame.sort_values(
            ["id", "_source_priority", "_completeness", "published"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates("id", keep="first")
        .copy()
    )

    for side in ("min", "max"):
        by_id[f"y_{side}_usd"] = pd.to_numeric(
            by_id[f"salary_{side}_usd"], errors="coerce"
        )
        by_id[f"{side}_is_reported"] = pd.to_numeric(
            by_id[f"salary_{side}"], errors="coerce"
        ).notna()

    by_id["target_source"] = np.select(
        [
            by_id["min_is_reported"] & by_id["max_is_reported"],
            by_id["min_is_reported"] ^ by_id["max_is_reported"],
            by_id["y_min_usd"].notna() & by_id["y_max_usd"].notna(),
        ],
        ["Reportado", "Híbrido", "Estimado"],
        default="Faltante",
    )

    url_norm = (
        by_id["apply_url"]
        .fillna("")
        .astype(str)
        .map(lambda value: urldefrag(value.strip().lower())[0].rstrip("/"))
    )
    company_norm = normalize_text(by_id["company"])
    title_norm = normalize_text(by_id["title"])
    location_norm = normalize_text(by_id["location"])
    key_parts = pd.DataFrame(
        {
            "url": url_norm,
            "company": company_norm,
            "title": title_norm,
            "location": location_norm,
        },
        index=by_id.index,
    )
    by_id["_vacancy_key"] = key_parts.agg("|".join, axis=1)
    empty_key = key_parts.eq("").all(axis=1)
    by_id.loc[empty_key, "_vacancy_key"] = "id:" + by_id.loc[empty_key, "id"].astype(str)

    source_rank = {"Reportado": 3, "Híbrido": 2, "Estimado": 1, "Faltante": 0}
    agency = parse_boolean(by_id["company_is_agency"]).fillna(False).astype(bool)
    by_id["_target_rank"] = by_id["target_source"].map(source_rank)
    by_id["_direct_rank"] = (~agency).astype(int)
    deduplicated = (
        by_id.sort_values(
            [
                "_vacancy_key",
                "_direct_rank",
                "_target_rank",
                "_completeness",
                "published",
            ],
            ascending=[True, False, False, False, False],
        )
        .drop_duplicates("_vacancy_key", keep="first")
        .copy()
    )

    has_target = deduplicated["y_min_usd"].notna() & deduplicated["y_max_usd"].notna()
    positive = has_target & (deduplicated["y_min_usd"] > 0) & (
        deduplicated["y_max_usd"] > 0
    )
    ordered = positive & (deduplicated["y_min_usd"] <= deduplicated["y_max_usd"])
    target = deduplicated.loc[ordered].copy()
    target["salary_midpoint_usd"] = (
        target["y_min_usd"] + target["y_max_usd"]
    ) / 2
    target["salary_width_usd"] = target["y_max_usd"] - target["y_min_usd"]

    log_midpoint = np.log10(target["salary_midpoint_usd"])
    q1, q3 = log_midpoint.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = float(10 ** (q1 - outlier_iqr_multiplier * iqr))
    upper = float(10 ** (q3 + outlier_iqr_multiplier * iqr))
    far_outlier = (target["salary_midpoint_usd"] < lower) | (
        target["salary_midpoint_usd"] > upper
    )
    model_frame = target.loc[~far_outlier].copy()

    audit: dict[str, object] = {
        "source_rows": int(source_rows),
        "unique_ids": int(len(by_id)),
        "duplicate_ids_removed": int(source_rows - len(by_id)),
        "unique_vacancies": int(len(deduplicated)),
        "republications_removed": int(len(by_id) - len(deduplicated)),
        "incomplete_targets": int((~has_target).sum()),
        "non_positive_targets": int((has_target & ~positive).sum()),
        "unordered_targets": int((positive & ~ordered).sum()),
        "far_outliers_removed": int(far_outlier.sum()),
        "outlier_lower_usd": lower,
        "outlier_upper_usd": upper,
        "model_rows": int(len(model_frame)),
        "target_source_counts": {
            str(key): int(value)
            for key, value in model_frame["target_source"].value_counts().items()
        },
        "date_min": _iso_or_none(model_frame["published"].min()),
        "date_max": _iso_or_none(model_frame["published"].max()),
    }
    return PreparationResult(frame=model_frame, audit=audit)


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    cutoffs_from: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict[str, str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction must leave a non-empty test fraction")

    reference = frame if cutoffs_from is None else cutoffs_from
    valid_dates = reference["published"].dropna()
    if valid_dates.empty:
        raise ValueError("Temporal split requires at least one valid publication date")
    train_cutoff = valid_dates.quantile(train_fraction)
    validation_cutoff = valid_dates.quantile(train_fraction + validation_fraction)

    split = pd.Series("test", index=frame.index, dtype="string")
    split.loc[frame["published"] <= validation_cutoff] = "validation"
    split.loc[frame["published"] <= train_cutoff] = "train"
    split.loc[frame["published"].isna()] = "excluded_missing_date"
    return split, {
        "train_cutoff": train_cutoff.isoformat(),
        "validation_cutoff": validation_cutoff.isoformat(),
    }


def select_experiment(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    normalized = experiment.strip().lower()
    if normalized == "reported":
        return frame.loc[frame["target_source"].eq("Reportado")].copy()
    if normalized == "expanded":
        return frame.copy()
    raise ValueError("data.experiment must be 'reported' or 'expanded'")


def source_file_manifest(files: list[Path]) -> list[dict[str, object]]:
    return [
        {"path": path.as_posix(), "size_bytes": int(path.stat().st_size)} for path in files
    ]


def _iso_or_none(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
