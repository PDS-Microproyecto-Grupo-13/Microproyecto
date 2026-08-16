# Foorilla Data Ingestion

Scope of this module: pull job posting / salary data from the Foorilla API,
validate it, and write it out as a dated CSV file for hand off.


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

notebooks/01_eda_ingested_data.ipynb    # starter notebook loading the latest jobs CSV
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
python -m src.ingestion.fetch --endpoint jobs # avoid this!!! 
python -m src.ingestion.fetch --endpoint jobs --topic 101 # Topic 101 = Data, AI and  Machine Learning
python -m src.ingestion.fetch --endpoint jobs --title "ml engineer" --location "remote"

# Quick test run — stop after a few pages instead of walking the whole catalog
python -m src.ingestion.fetch --endpoint jobs --max-pages 3

# Salary data
python -m src.ingestion.fetch --endpoint salaries
```

**⚠️ An unfiltered jobs pull is enormous.** As of 2026-08-15 `/hiring/job/`
with no filters returns ~3.4 million records (tens of thousands of pages).
**Filter by topic id instead** — see "Filtering by topic" below for how to
look up the right id. `--title` filtering (below) still works too, but topic filtering
is the better fit for scoping to "AI job positions".

```bash
python -m src.ingestion.fetch --endpoint jobs --title "machine learning engineer"
python -m src.ingestion.fetch --endpoint jobs --title "data scientist"
python -m src.ingestion.fetch --endpoint jobs --title "ai researcher"
```

This writes `data/raw/foorilla/<endpoint>_<date>.csv` plus a matching
`_manifest.json` (which includes the attribution notice — don't strip it
when handing files off) — **jobs rows are streamed to disk as they're
validated**, so you'll see partial progress on disk even mid-run, and a
`Ctrl+C` part-way through leaves you a valid (partial) CSV rather than
nothing. Whoever owns versioning picks these up next; that step isn't
handled in this module.

## Incremental pulls (catching up on new postings only)

Once you have an initial full pull done, use `--published-after` to only
fetch jobs published since then — much faster than re-pulling everything:

```bash
# Check what cutoff to use, based on your last pull's manifest
python -m src.ingestion.fetch --suggest-since

# Then pull just the new stuff
python -m src.ingestion.fetch --endpoint jobs --topic 101 --published-after 2026-08-15
```

**Caveat:** `published_after` filters on the job's `published` date, not on
when *you* pulled it — so if Foorilla backfills older postings into their
index after the fact, an incremental pull could still miss them. That's a
reasonable tradeoff for this project (small blind spot vs. hours saved),
but it's worth an occasional full re-pull (no `--published-after`) rather
than relying on incremental pulls forever.

Each pull writes its own dated file (`jobs_<pull-date>.csv`), so an
incremental pull won't overwrite your full historical one — whoever owns
data versioning will need to concatenate/dedupe across these dated files
rather than expecting one single "the data" file.

## Testing

```bash
pytest tests/unit -v
```

Tests mock all HTTP calls (`responses` library) — no real API key or network
access needed to run CI.

## Jobs schema 

`JOB_CSV_COLUMNS` / `JobRaw` are confirmed 2026-08-15 against the actual
Swagger/OpenAPI response schema for `GET /hiring/job/` — not just the
earlier manual CSV export, which turned out to differ in a few ways:

- **`is_agency` / `is_remote` live under `company`**, not top-level on the
  job. CSV column `company_is_agency` reflects this.
- **`views`, `clicks`, `foo_url`** (present in the dashboard CSV export)
  **don't appear in the API schema at all** — dropped from ingestion output
  since this module pulls from the API, not the export tool.
- **`topics` is embedded on every job record** — a list of
  `{id, name, is_main, tags}`. No separate lookup needed once you have job
  data; flattened into `topics` (pipe-separated names) and `topic_ids`.
- **Currency-normalized salary fields exist**: `salary_min_usd`/`_max_usd`
  and `_eur` variants, alongside original-currency `salary_min`/`_max`.
  **Use the USD/EUR columns for cross-country modeling** — comparing raw
  `salary_min` across currencies (INR vs. JPY vs. USD) isn't meaningful
  without conversion, and this endpoint already did that for you.

Full confirmed CSV output: `id`, `company`, `company_id`,
`company_is_agency`, `title`, `location`, `has_remote`, `work_mode`,
`published`, `expired`, `experience_level`, `experience_years`,
`language`, `salary_min`, `salary_max`, `salary_min_est`, `salary_max_est`,
`salary_currency`, `salary_min_usd`, `salary_max_usd`, `salary_min_eur`,
`salary_max_eur`, `topics`, `topic_ids`, `tags`, `regions`, `countries`,
`apply_url`.

## Filtering by topic 

`/hiring/job/`'s documented Parameters list confirms a real `topic`
filter — `array<integer>`, i.e. one or more topic **ids** (not names).
Full filter set available: `topic`, `tag`, `region`, `country` (all
integer id arrays), plus `title`, `location`, `company`,
`experience_level`, `language` (string partial-match),
`has_remote`/`company_remote`/`company_agency` (booleans), `work_mode`
(integer), and `published_after`/`published_before` (dates).

Since the UI shows topic *names* (e.g. "Data, AI, and Machine Learning")
but the API wants an *id*, look it up first:

```bash
python -m src.ingestion.fetch --find-topic "Data, AI"
# prints: id=<N>  name='Data, AI, and Machine Learning'
```

Then use that id to scope the actual pull — much better than an unfiltered
3.4M-record pull or guessing at title strings:

```bash
python -m src.ingestion.fetch --endpoint jobs --topic <N>
python -m src.ingestion.fetch --endpoint jobs --topic <N> --topic <M>   # multiple topics
```

Any other documented filter not yet wired up as its own flag can go
through `--extra KEY=VALUE` (repeatable), e.g. `--extra experience_level=senior`.

## ⚠️ Salaries endpoint still unconfirmed

`/insight/salary/`'s field names haven't been confirmed the way jobs' have.
`fetch_salaries()` currently infers CSV columns from whatever the first
record contains rather than a fixed list — get a sample record (dashboard
export or a live API call) before relying on this for real ingestion, and
update `SalaryRaw` in `schema.py` to match, mirroring what was done for jobs.
