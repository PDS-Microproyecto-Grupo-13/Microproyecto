from __future__ import annotations

import logging
import math
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from ml_pipeline.analytics_schema import AnalyticsSummary, PublishResponse
from ml_pipeline.common.io import read_json
from ml_pipeline.settings import Settings

LOGGER = logging.getLogger(__name__)


class PublishAnalyticsError(RuntimeError):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _publisher_config(settings: Settings) -> tuple[str, str, float]:
    url = (settings.analytics_publish_url or "").strip()
    token = settings.analytics_publish_token or ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PublishAnalyticsError(
            "ANALYTICS_PUBLISH_URL must be an absolute HTTP(S) URL", 3
        )
    if not token.strip():
        raise PublishAnalyticsError("ANALYTICS_PUBLISH_TOKEN is required", 3)
    try:
        timeout = float(settings.analytics_publish_timeout_seconds or "10")
    except (TypeError, ValueError) as error:
        raise PublishAnalyticsError(
            "ANALYTICS_PUBLISH_TIMEOUT_SECONDS must be numeric", 3
        ) from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise PublishAnalyticsError(
            "ANALYTICS_PUBLISH_TIMEOUT_SECONDS must be finite and greater than zero", 3
        )
    return url, token, timeout


def _load_summary(settings: Settings) -> AnalyticsSummary:
    path = settings.path("artifacts/reports/dashboard_summary.json")
    if not path.is_file():
        raise PublishAnalyticsError(
            f"Analytics artifact not found at {path}; run 'dvc repro analytics' first", 2
        )
    try:
        payload = read_json(path)
        return AnalyticsSummary.model_validate(payload)
    except (OSError, ValueError, ValidationError) as error:
        raise PublishAnalyticsError(
            "Analytics artifact is invalid; regenerate it with 'dvc repro analytics'", 2
        ) from error


def _safe_error_message(response: httpx.Response, token: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"Analytics backend returned HTTP {response.status_code}"
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        safe_message = payload["message"].replace(token, "<redacted>")
        return f"Analytics backend returned HTTP {response.status_code}: {safe_message}"
    return f"Analytics backend returned HTTP {response.status_code}"


def publish_analytics(
    settings: Settings,
    client: httpx.Client | None = None,
) -> PublishResponse:
    summary = _load_summary(settings)
    url, token, timeout = _publisher_config(settings)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    owns_client = client is None
    http_client = client or httpx.Client()
    try:
        try:
            response = http_client.post(
                url,
                headers=headers,
                json=summary.model_dump(mode="json"),
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as error:
            raise PublishAnalyticsError(
                "Unable to reach analytics backend due to a transport error", 6
            ) from error

        if 400 <= response.status_code < 500:
            raise PublishAnalyticsError(_safe_error_message(response, token), 4)
        if response.status_code >= 500:
            raise PublishAnalyticsError(_safe_error_message(response, token), 5)
        if response.status_code not in {200, 201}:
            raise PublishAnalyticsError(
                f"Analytics backend returned unexpected HTTP {response.status_code}", 5
            )
        try:
            result = PublishResponse.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise PublishAnalyticsError(
                "Analytics backend returned an invalid success response", 5
            ) from error
    finally:
        if owns_client:
            http_client.close()

    LOGGER.info(
        "publish-analytics | result=success | status=%s | artifact_hash=%s | stored_at=%s",
        result.status,
        result.artifact_hash,
        result.stored_at,
    )
    return result
