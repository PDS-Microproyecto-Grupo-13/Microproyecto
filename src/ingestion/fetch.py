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
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .client import FoorillaClient
from .config import RAW_DATA_DIR, get_config
from .schema import JobRaw, SalaryRaw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Confirmed 2026-08-15 against the real OpenAPI response schema for
# GET /hiring/job/ (Swagger docs at /api/v1/docs) — supersedes the earlier
# export-derived column list. `views`/`clicks`/`foo_url` from the manual
# CSV export are dropped: they don't appear in the API schema at all.
JOB_CSV_COLUMNS = [
    "id",
    "company",           # flattened company name (see _job_row)
    "company_id",        # for joining against /hiring/company/ later
    "company_is_agency",
    "title",
    "location",
    "has_remote",
    "work_mode",
    "published",
    "expired",
    "experience_level",
    "experience_years",
    "language",
    "salary_min",
    "salary_max",
    "salary_min_est",
    "salary_max_est",
    "salary_currency",
    "salary_min_usd",    # currency-normalized — use these for cross-country modeling
    "salary_max_usd",
    "salary_min_eur",
    "salary_max_eur",
    "topics",             # pipe-separated topic names, e.g. "Data, AI, and Machine Learning"
    "topic_ids",
    "tags",                # pipe-separated tag names
    "regions",              # pipe-separated region names
    "countries",             # pipe-separated country names
    "apply_url",
]

ATTRIBUTION_NOTICE = (
    "Data sourced from Foorilla (https://foorilla.com/api/), licensed under "
    "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)."
)


def _job_row(validated: JobRaw) -> dict[str, Any]:
    """Flatten a validated JobRaw into a CSV-ready row. Nested/list fields
    (company, topics, tags, regions, countries) are excluded from
    model_dump() and replaced with flattened scalar/pipe-joined values,
    since raw nested objects don't serialize sensibly to a flat CSV cell."""
    row = validated.model_dump(exclude={"company", "topics", "tags", "regions", "countries"})
    row["company"] = validated.company_name
    row["company_id"] = validated.company_id
    row["company_is_agency"] = validated.company_is_agency
    row["topics"] = validated.topic_names
    row["topic_ids"] = validated.topic_ids
    row["tags"] = validated.tag_names
    row["regions"] = validated.region_names
    row["countries"] = validated.country_names
    return row


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
    topic: int | list[int] | None = None,
    published_after: str | None = None,
    published_before: str | None = None,
    max_pages: int | None = None,
    extra_params: dict[str, Any] | None = None,
    pull_date: date | None = None,
) -> Path:
    config = get_config()
    client = FoorillaClient(config)
    pull_date = pull_date or datetime.now(timezone.utc).date()

    out_dir = RAW_DATA_DIR / "foorilla"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"jobs_{pull_date.isoformat()}.csv"
    manifest_path = out_dir / f"jobs_{pull_date.isoformat()}_manifest.json"

    n_seen = 0
    n_valid = 0
    n_invalid = 0

    # Stream rows to disk as they're validated instead of buffering the
    # whole pull in memory — for a multi-million-row pull, buffering means
    # nothing is saved (and nothing recoverable) until the entire run
    # finishes, which for an unfiltered pull can take many hours.
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOB_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for record in client.iter_jobs(
            title=title,
            location=location,
            company=company,
            topic=topic,
            published_after=published_after,
            published_before=published_before,
            max_pages=max_pages,
            extra_params=extra_params,
        ):
            n_seen += 1
            try:
                validated = JobRaw.from_api_record(record)
            except ValidationError as exc:
                n_invalid += 1
                logger.warning("Skipping invalid job record title=%r: %s", record.get("title"), exc)
                continue
            writer.writerow(_job_row(validated))
            n_valid += 1

            if n_valid % 500 == 0:
                f.flush()  # make progress visible on disk periodically, not just at EOF
                logger.info("Progress: %s valid rows written so far", n_valid)

    manifest = {
        "source": "foorilla",
        "endpoint": "hiring/job/",
        "license": "CC BY-SA 4.0",
        "attribution": ATTRIBUTION_NOTICE,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "title": title, "location": location, "company": company, "topic": topic,
            "published_after": published_after, "published_before": published_before,
            **(extra_params or {}),
        },
        "max_pages": max_pages,
        "records_seen": n_seen,
        "records_valid": n_valid,
        "records_invalid": n_invalid,
        "csv_path": str(csv_path.relative_to(RAW_DATA_DIR.parent)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Jobs ingestion complete: %s valid / %s seen (%s invalid). Output: %s",
        n_valid, n_seen, n_invalid, csv_path,
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


def find_topic_id(name_contains: str) -> None:
    """Look up topics by name and print their ids — since /hiring/job/'s
    `topic` filter takes an integer id, not a name. Prints matches to
    stdout rather than returning, since this is meant to be run as a
    one-off lookup from the command line.

    Matching is word-based, not literal-substring: "Data, AI, Machine
    Learning" (missing "and", different punctuation) still matches
    "Data, AI, and Machine Learning" because every *word* in the query
    appears somewhere in the topic name, regardless of exact commas/spacing.
    Pass an empty string to list every topic (there are few enough that
    this is a reasonable way to eyeball the full list).
    """
    config = get_config()
    client = FoorillaClient(config)

    words = [w for w in re.split(r"[^a-z0-9]+", name_contains.lower()) if w]
    found = False
    for record in client.iter_topics():
        name = record.get("name", "")
        name_lower = name.lower()
        if not words or all(w in name_lower for w in words):
            found = True
            print(f"id={record.get('id')}  name={name!r}")
    if not found:
        print(f"No topics found matching {name_contains!r}. Try a shorter search, "
              f"or run with --find-topic '' to list every topic.")


def suggest_since() -> None:
    """Scan existing dated jobs_*.csv manifests and print the most recent
    `pulled_at_utc` seen, as a starting point for --published-after on the
    next incremental pull. Doesn't guarantee zero gaps/overlap on its own —
    see README for the caveat about using `published` vs. pull date."""
    out_dir = RAW_DATA_DIR / "foorilla"
    manifests = sorted(out_dir.glob("jobs_*_manifest.json"))
    if not manifests:
        print("No previous jobs manifests found under data/raw/foorilla/ — nothing to suggest.")
        return

    latest = manifests[-1]
    data = json.loads(latest.read_text())
    print(f"Most recent manifest: {latest.name}")
    print(f"  pulled_at_utc: {data.get('pulled_at_utc')}")
    print(f"  filters used:  {data.get('filters')}")
    print(
        "\nSuggested next run — use the DATE portion of pulled_at_utc above as "
        "--published-after (or a day or two earlier, to be safe against any "
        "clock/timezone edge cases):"
    )
    print(f"  python -m src.ingestion.fetch --endpoint jobs --topic <id> "
          f"--published-after {data.get('pulled_at_utc', '')[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data from Foorilla.")
    parser.add_argument("--endpoint", choices=["jobs", "salaries"], default="jobs")
    parser.add_argument("--title", default=None, help="Partial match filter (jobs only)")
    parser.add_argument("--location", default=None, help="Partial match filter (jobs only)")
    parser.add_argument("--company", default=None, help="Partial match filter (jobs only)")
    parser.add_argument(
        "--topic",
        type=int,
        action="append",
        default=None,
        help="Filter by topic id (jobs only), repeatable for multiple topics. "
        "Don't know the id? Run: python -m src.ingestion.fetch --find-topic 'Data, AI'",
    )
    parser.add_argument(
        "--published-after",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only jobs published on/after this date (jobs only). Use for incremental "
        "catch-up pulls instead of re-pulling full history every time.",
    )
    parser.add_argument(
        "--published-before",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only jobs published before this date (jobs only).",
    )
    parser.add_argument(
        "--find-topic",
        default=None,
        metavar="NAME_SUBSTRING",
        help="Look up topic ids by (partial) name via /core/topic/ and print them, "
        "instead of running an ingestion pull. e.g. --find-topic 'Data, AI'",
    )
    parser.add_argument(
        "--suggest-since",
        action="store_true",
        help="Print a suggested --published-after value based on the most recent "
        "previous pull's manifest, instead of running an ingestion pull.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after this many pages (jobs only). Use for a quick test run — "
        "an unfiltered pull can be tens of thousands of pages.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra query filter to pass straight through to /hiring/job/, repeatable. "
        "For anything not already covered by a named flag above.",
    )
    args = parser.parse_args()

    if args.find_topic is not None:  # note: "is not None", not truthiness — "" means "list all"
        find_topic_id(args.find_topic)
        return

    if args.suggest_since:
        suggest_since()
        return

    extra_params: dict[str, Any] = {}
    for item in args.extra:
        if "=" not in item:
            parser.error(f"--extra must be KEY=VALUE, got: {item!r}")
        key, _, value = item.partition("=")
        extra_params[key] = value

    if args.endpoint == "jobs":
        fetch_jobs(
            title=args.title,
            location=args.location,
            company=args.company,
            topic=args.topic,
            published_after=args.published_after,
            published_before=args.published_before,
            max_pages=args.max_pages,
            extra_params=extra_params or None,
        )
    else:
        fetch_salaries()


if __name__ == "__main__":
    main()
