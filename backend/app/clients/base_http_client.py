import logging
import time
from typing import Any

import httpx

from app.core.exceptions import ExternalServiceError
from app.middleware.request_context import get_request_id

logger = logging.getLogger(__name__)


class BaseHttpClient:
    """Reusable asynchronous HTTP client with request ID propagation and centralized logging."""

    def __init__(
        self,
        service_name: str,
        base_url: str,
        timeout: float = 10.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = default_headers or {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize or return the persistent AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self.default_headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BaseHttpClient":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _prepare_headers(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        """Combines headers and injects X-Request-ID from context."""
        merged_headers = dict(self.default_headers)
        if headers:
            merged_headers.update(headers)

        request_id = get_request_id()
        if request_id:
            merged_headers["X-Request-ID"] = request_id

        return merged_headers

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_data: Any | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Executes an HTTP request with error translation, logging, and trace propagation."""
        client = await self._get_client()
        prepared_headers = self._prepare_headers(headers)
        request_id = get_request_id()
        formatted_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

        logger.info(
            "external_request_started",
            extra={
                "service": self.service_name,
                "request_id": request_id,
                "method": method.upper(),
                "endpoint": formatted_endpoint,
            },
        )

        start_time = time.perf_counter()

        try:
            response = await client.request(
                method=method.upper(),
                url=formatted_endpoint,
                params=params,
                json=json_data,
                headers=prepared_headers,
                timeout=timeout or self.timeout,
            )
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            logger.info(
                "external_request_completed",
                extra={
                    "service": self.service_name,
                    "request_id": request_id,
                    "method": method.upper(),
                    "endpoint": formatted_endpoint,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

            # Raise on 4xx/5xx status codes
            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "external_request_http_error",
                extra={
                    "service": self.service_name,
                    "request_id": request_id,
                    "endpoint": formatted_endpoint,
                    "status_code": exc.response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            raise ExternalServiceError(
                service_name=self.service_name,
                message=f"HTTP {exc.response.status_code} received from upstream service",
                status_code=exc.response.status_code if exc.response.status_code < 500 else 502,
                details={"response_text": exc.response.text},
            ) from exc

        except httpx.TimeoutException as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "external_request_timeout",
                extra={
                    "service": self.service_name,
                    "request_id": request_id,
                    "endpoint": formatted_endpoint,
                    "duration_ms": duration_ms,
                },
            )
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Request timed out while contacting upstream service",
                status_code=504,
            ) from exc

        except httpx.RequestError as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "external_request_connection_error",
                extra={
                    "service": self.service_name,
                    "request_id": request_id,
                    "endpoint": formatted_endpoint,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                },
            )
            raise ExternalServiceError(
                service_name=self.service_name,
                message=f"Connection failed: {exc}",
                status_code=502,
            ) from exc

    async def get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Helper for HTTP GET."""
        return await self.request("GET", endpoint, params=params, headers=headers, timeout=timeout)

    async def post(
        self,
        endpoint: str,
        *,
        json_data: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Helper for HTTP POST."""
        return await self.request(
            "POST",
            endpoint,
            params=params,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
        )
