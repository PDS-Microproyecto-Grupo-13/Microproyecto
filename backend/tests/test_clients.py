import httpx
import pytest

from app.clients.base_http_client import BaseHttpClient
from app.core.exceptions import ExternalServiceError
from app.middleware.request_context import _request_id_ctx_var


@pytest.mark.asyncio
async def test_base_http_client_propagates_request_id() -> None:
    """Test that BaseHttpClient automatically injects the active X-Request-ID."""
    token = _request_id_ctx_var.set("client-trace-id-abc")

    captured_headers: dict[str, str] = {}

    def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = dict(request.headers)
        return httpx.Response(200, json={"status": "upstream_ok"})

    transport = httpx.MockTransport(mock_transport_handler)

    client = BaseHttpClient(service_name="test-service", base_url="http://upstream.local")
    client._client = httpx.AsyncClient(transport=transport, base_url="http://upstream.local")

    try:
        response = await client.get("/test-endpoint")
        assert response.status_code == 200
        assert captured_headers.get("x-request-id") == "client-trace-id-abc"
    finally:
        _request_id_ctx_var.reset(token)
        await client.close()


@pytest.mark.asyncio
async def test_base_http_client_translates_http_error() -> None:
    """Test that BaseHttpClient translates upstream HTTP 500 into ExternalServiceError."""

    def mock_error_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal upstream error")

    transport = httpx.MockTransport(mock_error_handler)

    client = BaseHttpClient(service_name="inference-service", base_url="http://inference.local")
    client._client = httpx.AsyncClient(transport=transport, base_url="http://inference.local")

    try:
        with pytest.raises(ExternalServiceError) as exc_info:
            await client.post("/predict", json_data={"features": [1, 2, 3]})

        assert exc_info.value.service_name == "inference-service"
        assert exc_info.value.status_code == 502
    finally:
        await client.close()
