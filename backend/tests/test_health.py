import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.main import create_app
from app.schemas.health import HealthResponse


@pytest.mark.asyncio
async def test_health_returns_200_and_valid_schema(
    client: AsyncClient, test_settings: Settings
) -> None:
    """Test that GET /api/v1/health returns HTTP 200 and matches HealthResponse schema."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    # Validate against Pydantic schema
    health_response = HealthResponse(**data)
    assert health_response.status == "ok"
    assert health_response.service == test_settings.APP_NAME
    assert health_response.version == test_settings.APP_VERSION


@pytest.mark.asyncio
async def test_health_generates_request_id_header(client: AsyncClient) -> None:
    """Test that X-Request-ID header is generated and returned when not provided by client."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert len(request_id) > 0


@pytest.mark.asyncio
async def test_health_propagates_custom_request_id(client: AsyncClient) -> None:
    """Test that client-provided X-Request-ID header is propagated back in response."""
    custom_request_id = "test-custom-trace-uuid-12345"
    response = await client.get(
        "/api/v1/health",
        headers={"X-Request-ID": custom_request_id},
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_request_id


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500_with_request_id(test_settings: Settings) -> None:
    """Test that unhandled errors return HTTP 500 without leaking stack traces."""
    app = create_app(settings=test_settings)

    @app.get("/api/v1/error-trigger")
    async def trigger_error():
        raise RuntimeError("Simulated internal runtime failure")

    # raise_app_exceptions=False lets Starlette ServerErrorMiddleware return HTTP 500 response
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        response = await test_client.get(
            "/api/v1/error-trigger",
            headers={"X-Request-ID": "err-trace-999"},
        )

        assert response.status_code == 500

        data = response.json()
        assert data["error"] == "internal_server_error"
        assert data["message"] == "An unexpected error occurred"
        assert data["request_id"] == "err-trace-999"
        # Ensure stack trace is not in JSON response
        assert "traceback" not in data
        assert "RuntimeError" not in data["message"]


@pytest.mark.asyncio
async def test_application_error_handling(test_settings: Settings) -> None:
    """Test that custom ApplicationError exceptions are handled with proper status codes."""
    app = create_app(settings=test_settings)

    @app.get("/api/v1/custom-error")
    async def trigger_custom_error():
        raise ApplicationError(
            message="Resource not found in cache",
            status_code=404,
            error_code="not_found",
        )

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        response = await test_client.get("/api/v1/custom-error")

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "not_found"
        assert data["message"] == "Resource not found in cache"
        assert "request_id" in data
