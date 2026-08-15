# Foorilla Data Ingestion

Scope of this module: pull job posting / salary data from the Foorilla API,
validate it, and write it out as a dated CSV file that's easy to hand off.

## API reference

- Base URL: `https://foorilla.com/api/v1/`
- Auth: `Api-Key: YOUR_API_KEY` header
- Rate limit: 5 requests/second (binding constraint; also capped at 600/min)
- Docs: https://foorilla.com/api/v1/docs · Schema: https://foorilla.com/api/v1/schema.json
- **License: CC BY-SA 4.0 — attribution required.** See `ATTRIBUTION.md`.

## Layout

```
src/ingestion/
├── config.py     # env-based config (API key, rate limit, etc.)
├── client.py     # Foorilla API client: auth, pagination, retries, throttling
├── schema.py     # pydantic validation of raw records (field names pending confirmation)
└── fetch.py      # orchestrates client -> validation -> dated CSV

data/raw/foorilla/
├── jobs_2026-08-15.csv                  # from /hiring/job/
├── jobs_2026-08-15_manifest.json        # counts, filters used, attribution
├── salaries_2026-08-15.csv              # from /insight/salary/
└── salaries_2026-08-15_manifest.json

notebooks/01_eda_ingested_data.ipynb   # starter notebook loading the latest jobs CSV
tests/unit/test_client.py               # mocked HTTP tests, no real API calls
ATTRIBUTION.md                          # CC BY-SA 4.0 attribution notice — read before publishing anything
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running an ingestion pull

```bash
# Job postings (default), optionally filtered
python -m src.ingestion.fetch --endpoint jobs
python -m src.ingestion.fetch --endpoint jobs --title "ml engineer" --location "remote"

# Salary data
python -m src.ingestion.fetch --endpoint salaries
```

This writes `data/raw/foorilla/<endpoint>_<date>.csv` plus a matching
`_manifest.json` (which includes the attribution notice — don't strip it
when handing files off). Whoever owns versioning picks these up next; that
step isn't handled in this module.

## Testing

```bash
pytest tests/unit -v
```

Tests mock all HTTP calls (`responses` library) — no real API key or network
access needed to run CI.

## Jobs schema

`JOB_CSV_COLUMNS` / `JobRaw` 

`company`, `title`, `location`, `has_remote`, `is_agency`, `published`,
`expired`, `experience_level`, `experience_years`, `salary_min`,
`salary_max`, `salary_currency`, `views`, `clicks`, `foo_url`, `apply_url`


