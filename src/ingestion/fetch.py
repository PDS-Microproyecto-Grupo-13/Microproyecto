"""Entry point: fetch data from Foorilla and land it as a dated CSV file.

Usage:
    python -m src.ingestion.fetch --endpoint jobs
    python -m src.ingestion.fetch --endpoint jobs --title "ml engineer" --location "remote"
    python -m src.ingestion.fetch --endpoint salaries

Output layout (one CSV per pull date + endpoint):

    data/raw/foorilla/jobs_2026-08-15.csv
    data/raw/foorilla/jobs_2026-08-15_manifest.json
    data/raw/foorilla/salaries_2026-08-15.csv
    data/raw/foorilla/salaries_2026-08-15_manifest.json

LICENSE NOTE: Foorilla's API data is licensed CC BY-SA 4.0, which requires
attribution and "share-alike" (adapted/derived data must carry the same
license). The manifest written alongside each CSV includes attribution
metadata for this reason — don't strip it when handing files off. See
ATTRIBUTION.md at the repo root for the full notice to include in any
publication, dashboard, or report that uses this data.

This module's responsibility ends at producing these CSVs. Handing them
off to whoever owns data versioning is a separate step, not done here.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .client import FoorillaClient
from .config import RAW_DATA_DIR, get_config
from .schema import JobRaw, SalaryRaw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Confirmed 2026-08-15 against a manual CSV export from Foorilla's dashboard.
# Note: this is the *dashboard export* shape — the raw JSON from /hiring/job/
# may include an `id` or other fields not shown in the CSV export view.
# extra="allow" on JobRaw means those won't be lost if present, just not
# written as their own CSV column unless added here.
JOB_CSV_COLUMNS = [
    "company",
    "title",
    "location",
    "has_remote",
    "is_agency",
    "published",
    "expired",
    "experience_level",
    "experience_years",
    "salary_min",
    "salary_max",
    "salary_currency",
    "views",
    "clicks",
    "foo_url",
    "apply_url",
]

ATTRIBUTION_NOTICE = (
    "Data sourced from Foorilla (https://foorilla.com/api/), licensed under "
    "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)."
)


def _write_csv(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fetch_jobs(
    *,
    title: str | None = None,
    location: str | None = None,
    company: str | None = None,
    pull_date: date | None = None,
) -> Path:
    config = get_config()
    client = FoorillaClient(config)
    pull_date = pull_date or datetime.now(timezone.utc).date()

    out_dir = RAW_DATA_DIR / "foorilla"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"jobs_{pull_date.isoformat()}.csv"
    manifest_path = out_dir / f"jobs_{pull_date.isoformat()}_manifest.json"

    valid_rows: list[dict] = []
    n_seen = 0
    n_invalid = 0

    for record in client.iter_jobs(title=title, location=location, company=company):
        n_seen += 1
        try:
            validated = JobRaw.from_api_record(record)
        except ValidationError as exc:
            n_invalid += 1
            logger.warning("Skipping invalid job record title=%r: %s", record.get("title"), exc)
            continue
        valid_rows.append(validated.model_dump())

    _write_csv(valid_rows, JOB_CSV_COLUMNS, csv_path)

    manifest = {
        "source": "foorilla",
        "endpoint": "hiring/job/",
        "license": "CC BY-SA 4.0",
        "attribution": ATTRIBUTION_NOTICE,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {"title": title, "location": location, "company": company},
        "records_seen": n_seen,
        "records_valid": len(valid_rows),
        "records_invalid": n_invalid,
        "csv_path": str(csv_path.relative_to(RAW_DATA_DIR.parent)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Jobs ingestion complete: %s valid / %s seen (%s invalid). Output: %s",
        len(valid_rows), n_seen, n_invalid, csv_path,
    )
    return csv_path


def fetch_salaries(*, pull_date: date | None = None) -> Path:
    config = get_config()
    client = FoorillaClient(config)
    pull_date = pull_date or datetime.now(timezone.utc).date()

    out_dir = RAW_DATA_DIR / "foorilla"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"salaries_{pull_date.isoformat()}.csv"
    manifest_path = out_dir / f"salaries_{pull_date.isoformat()}_manifest.json"

    valid_rows: list[dict] = []
    n_seen = 0
    n_invalid = 0

    for record in client.iter_salaries():
        n_seen += 1
        try:
            validated = SalaryRaw.from_api_record(record)
        except ValidationError as exc:
            n_invalid += 1
            logger.warning("Skipping invalid salary record id=%s: %s", record.get("id"), exc)
            continue
        valid_rows.append(validated.model_dump())

    # Column set for /insight/salary/ is unconfirmed — write whatever keys
    # showed up on the first record rather than a hardcoded list, until
    # the real schema is confirmed and this can be tightened like jobs above.
    columns = list(valid_rows[0].keys()) if valid_rows else ["id"]
    _write_csv(valid_rows, columns, csv_path)

    manifest = {
        "source": "foorilla",
        "endpoint": "insight/salary/",
        "license": "CC BY-SA 4.0",
        "attribution": ATTRIBUTION_NOTICE,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_seen": n_seen,
        "records_valid": len(valid_rows),
        "records_invalid": n_invalid,
        "csv_path": str(csv_path.relative_to(RAW_DATA_DIR.parent)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Salaries ingestion complete: %s valid / %s seen (%s invalid). Output: %s",
        len(valid_rows), n_seen, n_invalid, csv_path,
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data from Foorilla.")
    parser.add_argument("--endpoint", choices=["jobs", "salaries"], default="jobs")
    parser.add_argument("--title", default=None, help="Partial match filter (jobs only)")
    parser.add_argument("--location", default=None, help="Partial match filter (jobs only)")
    parser.add_argument("--company", default=None, help="Partial match filter (jobs only)")
    args = parser.parse_args()

    if args.endpoint == "jobs":
        fetch_jobs(title=args.title, location=args.location, company=args.company)
    else:
        fetch_salaries()


if __name__ == "__main__":
    main()
