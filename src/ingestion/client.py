"""Client for the Foorilla API (v1).

Reference: https://foorilla.com/api/v1/docs

Handles: Api-Key auth, rate limiting (5 req/sec cap), pagination
(page/pages/count), and retry/backoff on transient failures.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import FoorillaConfig

logger = logging.getLogger(__name__)


class FoorillaAPIError(Exception):
    """Raised for non-retryable API errors (4xx other than 429)."""


class FoorillaClient:
    def __init__(self, config: FoorillaConfig):
        self.config = config
        self.session = self._build_session()
        self._min_interval = 1.0 / max(config.requests_per_second, 0.1)
        self._last_request_ts = 0.0

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        # Foorilla auth: "Api-Key: YOUR_API_KEY" header (not Bearer auth).
        session.headers.update(
            {
                "Api-Key": self.config.api_key,
                "Accept": "application/json",
            }
        )
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _throttle(self) -> None:
        """Client-side pacing to stay under 5 req/sec, on top of urllib3's
        reactive retry-on-429 — avoids tripping the limit in the first place."""
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        url = f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.config.timeout_seconds)

        if response.status_code == 401:
            raise FoorillaAPIError("Authentication failed — check FOORILLA_API_KEY / Api-Key header.")
        if response.status_code == 403:
            raise FoorillaAPIError("Forbidden — check subscription tier / scopes for this endpoint.")
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise FoorillaAPIError(f"Client error {response.status_code}: {response.text[:500]}")

        response.raise_for_status()
        return response.json()

    def iter_paginated(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Generic paginator for any Foorilla list endpoint.

        Foorilla's paginated response shape:
            {"results": [...], "count": int, "pages": int, "page": int, "page_size": int}

        Yields one record at a time so callers can stream to disk.

        `max_pages` stops early after that many pages — useful for a quick
        test run (or a deliberately partial pull) instead of walking the
        entire result set.
        """
        query: dict[str, Any] = {"page_size": self.config.page_size, **(params or {})}

        page = 1
        total_pages: int | None = None
        while True:
            query["page"] = page
            logger.info("Fetching %s page=%s/%s params=%s", endpoint, page, total_pages or "?", query)
            payload = self._get(endpoint, params=query)

            results = payload.get("results", [])
            if not results:
                break
            yield from results

            total_pages = payload.get("pages")
            if total_pages is None or page >= total_pages:
                break
            if max_pages is not None and page >= max_pages:
                logger.info("Stopping early at max_pages=%s (out of %s total)", max_pages, total_pages)
                break
            page += 1

    def iter_jobs(
        self,
        *,
        title: str | None = None,
        location: str | None = None,
        company: str | None = None,
        topic: int | list[int] | None = None,
        tag: int | list[int] | None = None,
        region: int | list[int] | None = None,
        country: int | list[int] | None = None,
        experience_level: str | None = None,
        language: str | None = None,
        has_remote: bool | None = None,
        work_mode: int | None = None,
        company_remote: bool | None = None,
        company_agency: bool | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        extra_params: dict[str, Any] | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield job postings from /hiring/job/, using the full documented
        filter set. `topic`/`tag`/`region`/`country` take integer ID(s)
        (single int or a list) — not names; look IDs up via /core/topic/
        etc. first."""
        params: dict[str, Any] = {**(extra_params or {})}
        if title:
            params["title"] = title
        if location:
            params["location"] = location
        if company:
            params["company"] = company
        if topic is not None:
            params["topic"] = topic if isinstance(topic, list) else [topic]
        if tag is not None:
            params["tag"] = tag if isinstance(tag, list) else [tag]
        if region is not None:
            params["region"] = region if isinstance(region, list) else [region]
        if country is not None:
            params["country"] = country if isinstance(country, list) else [country]
        if experience_level:
            params["experience_level"] = experience_level
        if language:
            params["language"] = language
        if has_remote is not None:
            params["has_remote"] = has_remote
        if work_mode is not None:
            params["work_mode"] = work_mode
        if company_remote is not None:
            params["company_remote"] = company_remote
        if company_agency is not None:
            params["company_agency"] = company_agency
        if published_after:
            params["published_after"] = published_after
        if published_before:
            params["published_before"] = published_before
        yield from self.iter_paginated("hiring/job/", params=params, max_pages=max_pages)

    def iter_topics(self, *, extra_params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Yield records from /core/topic/ — used to look up a topic's
        integer id by name before filtering jobs with it."""
        yield from self.iter_paginated("core/topic/", params=extra_params or {})

    def iter_salaries(self, *, extra_params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Yield records from /insight/salary/ (the Insight space's salary data)."""
        yield from self.iter_paginated("insight/salary/", params=extra_params or {})
