from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Fixture providing test configuration settings."""
    return Settings(
        APP_NAME="mlops-backend-test",
        APP_VERSION="0.1.0-test",
        ENVIRONMENT="test",
        API_PREFIX="/api/v1",
        LOG_LEVEL="DEBUG",
        LOG_FORMAT="console",
        ANALYTICS_PUBLISH_TOKEN="test-analytics-token",
        ANALYTICS_STORAGE_PATH=tmp_path / "analytics_summary.json",
    )


@pytest.fixture
async def client(test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP test client bound to the FastAPI test application instance."""
    app = create_app(settings=test_settings)

    async def override_settings() -> Settings:
        return test_settings

    app.dependency_overrides[get_settings] = override_settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client

    app.dependency_overrides.clear()
