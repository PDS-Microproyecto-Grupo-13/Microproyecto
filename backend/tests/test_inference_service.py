import json

import httpx
import pytest

from app.core.exceptions import (
    InferenceResponseError,
    InferenceTimeoutError,
    InferenceUnavailableError,
)
from app.middleware.request_context import _request_id_ctx_var
from app.services.inference_service import InferenceService


@pytest.mark.asyncio
async def test_invoke_posts_payload_and_returns_json_object() -> None:
    captured_request: httpx.Request | None = None
    request_payload = {"arbitrary_input": {"values": [1, "two", True]}}
    provider_payload = {"predictions": [123.45], "metadata": {"model": "opaque"}}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(200, json=provider_payload)

    transport = httpx.MockTransport(handler)
    token = _request_id_ctx_var.set("inference-trace-123")
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://inference.local"
        ) as http_client:
            service = InferenceService(
                base_url="http://inference.local",
                timeout_seconds=10.0,
                http_client=http_client,
            )

            response_payload = await service.invoke(request_payload)
    finally:
        _request_id_ctx_var.reset(token)

    assert response_payload == provider_payload
    assert captured_request is not None
    assert captured_request.method == "POST"
    assert captured_request.url.path == "/invocations"
    assert captured_request.headers["content-type"] == "application/json"
    assert captured_request.headers["x-request-id"] == "inference-trace-123"
    assert json.loads(captured_request.content) == request_payload


@pytest.mark.asyncio
async def test_invoke_translates_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider timed out", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inference.local"
    ) as http_client:
        service = InferenceService("http://inference.local", 10.0, http_client)

        with pytest.raises(InferenceTimeoutError):
            await service.invoke({"input": "value"})


@pytest.mark.asyncio
async def test_invoke_translates_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inference.local"
    ) as http_client:
        service = InferenceService("http://inference.local", 10.0, http_client)

        with pytest.raises(InferenceUnavailableError):
            await service.invoke({"input": "value"})


@pytest.mark.asyncio
async def test_invoke_translates_provider_http_error() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(500, text="sensitive provider details")
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inference.local"
    ) as http_client:
        service = InferenceService("http://inference.local", 10.0, http_client)

        with pytest.raises(InferenceResponseError) as exc_info:
            await service.invoke({"input": "value"})

    assert "sensitive provider details" not in exc_info.value.message


@pytest.mark.asyncio
async def test_invoke_rejects_non_json_provider_response() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text="not-json"))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inference.local"
    ) as http_client:
        service = InferenceService("http://inference.local", 10.0, http_client)

        with pytest.raises(InferenceResponseError):
            await service.invoke({"input": "value"})


@pytest.mark.asyncio
async def test_invoke_rejects_json_response_that_is_not_an_object() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=[1, 2, 3]))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://inference.local"
    ) as http_client:
        service = InferenceService("http://inference.local", 10.0, http_client)

        with pytest.raises(InferenceResponseError):
            await service.invoke({"input": "value"})
