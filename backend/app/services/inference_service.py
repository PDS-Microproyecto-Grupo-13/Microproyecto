import logging
import time
from types import TracebackType
from typing import Any, Self

import httpx

from app.core.exceptions import (
    InferenceResponseError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from app.middleware.request_context import get_request_id

logger = logging.getLogger(__name__)

_INVOCATIONS_PATH = "/invocations"


class InferenceService:
    """HTTP adapter for the external inference service contract."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Content-Type": "application/json"},
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the internally managed HTTP connection pool."""
        if self._owns_http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a payload to the provider and return its JSON object response."""
        request_id = get_request_id()
        start_time = time.perf_counter()
        logger.info(
            "inference_request",
            extra={"request_id": request_id, "endpoint": _INVOCATIONS_PATH},
        )

        try:
            response = await self._http_client.post(
                _INVOCATIONS_PATH,
                json=payload,
                headers=self._request_headers(request_id),
            )
        except httpx.TimeoutException as exc:
            self._log_failure(start_time, reason="timeout")
            raise InferenceTimeoutError from exc
        except httpx.RequestError as exc:
            self._log_failure(start_time, reason="unavailable")
            raise InferenceUnavailableError from exc

        if response.is_error:
            self._log_failure(
                start_time,
                reason="provider_http_error",
                status_code=response.status_code,
            )
            raise InferenceResponseError

        try:
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                raise TypeError("Provider response must be a JSON object")
        except (TypeError, ValueError) as exc:
            self._log_failure(
                start_time,
                reason="invalid_provider_response",
                status_code=response.status_code,
            )
            raise InferenceResponseError from exc

        logger.info(
            "inference_success",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": self._duration_ms(start_time),
            },
        )
        return response_payload

    @staticmethod
    def _request_headers(request_id: str | None) -> dict[str, str] | None:
        return {"X-Request-ID": request_id} if request_id else None

    def _log_failure(
        self,
        start_time: float,
        *,
        reason: str,
        status_code: int | None = None,
    ) -> None:
        logger.error(
            "inference_failed",
            extra={
                "request_id": get_request_id(),
                "reason": reason,
                "status_code": status_code,
                "duration_ms": self._duration_ms(start_time),
            },
        )

    @staticmethod
    def _duration_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 2)
