from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

import numpy as np
import pandas as pd

from ml_pipeline.common.io import ensure_parent, sha256_file, write_json
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Raised when snapshot requirements before collect are not satisfied."""


def preflight_snapshots(source_dir: Path) -> list[Path]:
    """Inspect source directory to ensure all DVC-tracked snapshots are materialized."""
    if not source_dir.is_dir():
        raise PreflightError(f"Raw source directory does not exist: {source_dir}")

    # Inspect all jobs_*.csv.dvc pointers
    dvc_files = sorted(source_dir.glob("jobs_*.csv.dvc"))
    missing: list[str] = []
    for dvc_path in dvc_files:
        csv_name = dvc_path.stem  # e.g., 'jobs_2026-08-16.csv' from 'jobs_2026-08-16.csv.dvc'
        csv_path = source_dir / csv_name
        if not csv_path.is_file():
            missing.append(csv_name)

    if missing:
        missing_dvc_files = " ".join(f"{source_dir.as_posix()}/{name}.dvc" for name in missing)
        raise PreflightError(
            f"Missing {len(missing)} materialized snapshot(s) tracked by DVC in {source_dir}: {', '.join(missing)}. "
            f"Silent partial dataset processing is not permitted. "
            f"Please run the following command to download them before proceeding:\n"
            f"  dvc pull {missing_dvc_files}"
        )

    csv_files = sorted(source_dir.glob("jobs_*.csv"))
    if not csv_files:
        raise PreflightError(f"No materialized jobs_*.csv snapshot files found in {source_dir}")

    return csv_files


def normalize_text(values: pd.Series) -> pd.Series:
    """Normalize textual series according to canonical pipeline cleaning rules."""
    return (
        values.fillna("")
        .astype(str)
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def build_signature(frame: pd.DataFrame) -> pd.Series:
    """Construct deduplication signature from URL, company, title, and location."""
    apply_url = frame["apply_url"].fillna("").astype(str) if "apply_url" in frame.columns else pd.Series("", index=frame.index)
    url = apply_url.map(lambda val: urldefrag(str(val))[0].rstrip("/").lower())
    company = normalize_text(frame["company"]) if "company" in frame.columns else pd.Series("", index=frame.index)
    title = normalize_text(frame["title"]) if "title" in frame.columns else pd.Series("", index=frame.index)
    location = normalize_text(frame["location"]) if "location" in frame.columns else pd.Series("", index=frame.index)
    return url + "|" + company + "|" + title + "|" + location


def deduplicate_by_id(raw: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate records by ID, keeping the latest/most complete snapshot record."""
    return (
        raw.sort_values(["id", "_priority", "_complete", "published"])
        .drop_duplicates("id", keep="last")
        .copy()
    )


def deduplicate_by_signature(by_id: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate repostings by signature preserving the most informative record."""
    salary_cols = ["salary_min", "salary_max", "salary_min_usd", "salary_max_usd"]
    available_salary_cols = [col for col in salary_cols if col in by_id.columns]
    by_id["_salary_info"] = by_id[available_salary_cols].notna().sum(axis=1) if available_salary_cols else 0
    by_id["_signature"] = build_signature(by_id)

    if "company_is_agency" not in by_id.columns:
        by_id["company_is_agency"] = False

    return (
        by_id.sort_values([
            "_signature",
            "company_is_agency",
            "_salary_info",
            "_complete",
            "published",
        ])
        .drop_duplicates("_signature", keep="last")
        .copy()
    )


def compute_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract and categorize continuous target salaries and their reporting source."""
    df = frame.copy()
    for side in ["min", "max"]:
        usd_col = f"salary_{side}_usd"
        orig_col = f"salary_{side}"
        df[f"y_{side}_usd"] = (
            pd.to_numeric(df[usd_col], errors="coerce")
            if usd_col in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        df[f"{side}_reported"] = (
            pd.to_numeric(df[orig_col], errors="coerce").notna()
            if orig_col in df.columns
            else pd.Series(False, index=df.index)
        )

    df["target_source"] = np.select(
        [
            df["min_reported"] & df["max_reported"],
            df["min_reported"] | df["max_reported"],
        ],
        ["reportado", "híbrido"],
        default="estimado",
    )
    return df


def filter_valid_and_inliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter records by target hygiene (completeness, positivity, order) and IQR inliers."""
    complete = frame["y_min_usd"].notna() & frame["y_max_usd"].notna()
    positive = complete & (frame["y_min_usd"] > 0) & (frame["y_max_usd"] > 0)
    ordered = positive & (frame["y_min_usd"] <= frame["y_max_usd"])
    candidate = frame.loc[ordered].copy()

    stats = {
        "y_incompleta": int((~complete).sum()),
        "y_no_positiva": int((complete & ~positive).sum()),
        "y_invertida": int((positive & ~ordered).sum()),
    }

    if candidate.empty:
        stats["extremos_retirados"] = 0
        return candidate, stats

    midpoint = (candidate["y_min_usd"] + candidate["y_max_usd"]) / 2
    log_mid = np.log1p(midpoint)
    q1 = float(log_mid.quantile(0.25))
    q3 = float(log_mid.quantile(0.75))
    iqr = q3 - q1
    inlier = log_mid.between(q1 - 3 * iqr, q3 + 3 * iqr)
    stats["extremos_retirados"] = int((~inlier).sum())

    interim = candidate.loc[inlier].copy()
    return interim, stats


def create_data_manifest(csv_files: list[Path], raw_row_counts: dict[str, int]) -> dict[str, Any]:
    """Generate deterministic data manifest representing participating snapshots."""
    snapshots: list[dict[str, Any]] = []
    tokens: list[str] = []

    for path in sorted(csv_files, key=lambda p: p.name):
        fingerprint = sha256_file(path) or ""
        dvc_file = path.with_name(f"{path.name}.dvc")
        snapshots.append({
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "fingerprint": fingerprint,
            "rows_raw": raw_row_counts.get(path.name, 0),
            "dvc_tracked": dvc_file.is_file(),
        })
        tokens.append(f"{path.name}:{fingerprint}")

    dataset_fingerprint = hashlib.sha256(";".join(tokens).encode("utf-8")).hexdigest()
    return {
        "source": "foorilla",
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "dataset_fingerprint": dataset_fingerprint,
    }


def collect(settings: Settings) -> None:
    source_dir = settings.path("data/raw/foorilla")
    interim_output = settings.path("data/interim/foorilla_consolidated.parquet")
    manifest_output = settings.path("artifacts/reports/data_manifest.json")

    LOGGER.info("collect | preflight_check | source_dir=%s", source_dir)
    csv_files = preflight_snapshots(source_dir)
    LOGGER.info("collect | snapshots_found=%d", len(csv_files))

    parts: list[pd.DataFrame] = []
    raw_row_counts: dict[str, int] = {}
    for priority, path in enumerate(csv_files):
        LOGGER.info("collect | reading_snapshot priority=%d file=%s", priority, path.name)
        frame = pd.read_csv(path, low_memory=False)
        frame["_priority"] = priority
        frame["_source_file"] = path.name
        raw_row_counts[path.name] = len(frame)
        parts.append(frame)

    raw = pd.concat(parts, ignore_index=True)
    raw["published"] = pd.to_datetime(raw["published"], errors="coerce", utc=True)
    raw["_complete"] = raw.notna().sum(axis=1)
    filas_integradas = len(raw)

    by_id = deduplicate_by_id(raw)
    id_unicos = len(by_id)

    dedup = deduplicate_by_signature(by_id)
    vacantes_deduplicadas = len(dedup)

    with_targets = compute_targets(dedup)
    interim_df, filter_stats = filter_valid_and_inliers(with_targets)
    filas_finales = len(interim_df)

    source_counts = {
        category: int((interim_df["target_source"] == category).sum())
        for category in ["reportado", "híbrido", "estimado"]
    }

    LOGGER.info(
        "collect | statistics | snapshots=%d | filas_integradas=%d | id_unicos=%d | "
        "vacantes_deduplicadas=%d | y_incompleta=%d | y_no_positiva=%d | y_invertida=%d | "
        "extremos_retirados=%d | filas_finales=%d | target_sources=%s",
        len(csv_files),
        filas_integradas,
        id_unicos,
        vacantes_deduplicadas,
        filter_stats["y_incompleta"],
        filter_stats["y_no_positiva"],
        filter_stats["y_invertida"],
        filter_stats["extremos_retirados"],
        filas_finales,
        source_counts,
    )

    ensure_parent(interim_output)
    interim_df.to_parquet(interim_output, index=False, engine="pyarrow")
    LOGGER.info("collect | saved_interim | path=%s | rows=%d", interim_output, filas_finales)

    manifest = create_data_manifest(csv_files, raw_row_counts)
    write_json(manifest_output, manifest)
    LOGGER.info(
        "collect | saved_manifest | path=%s | dataset_fingerprint=%s",
        manifest_output,
        manifest["dataset_fingerprint"],
    )
