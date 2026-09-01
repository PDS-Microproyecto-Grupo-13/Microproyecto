from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing test configuration settings."""
    return Settings(
        APP_NAME="mlops-backend-test",
        APP_VERSION="0.1.0-test",
        ENVIRONMENT="test",
        API_PREFIX="/api/v1",
        LOG_LEVEL="DEBUG",
        LOG_FORMAT="console",
    )


@pytest.fixture
def breast_cancer_payload() -> dict[str, float]:
    """Valid public request payload using one sklearn Breast Cancer sample."""
    return {
        "mean_radius": 17.99,
        "mean_texture": 10.38,
        "mean_perimeter": 122.8,
        "mean_area": 1001.0,
        "mean_smoothness": 0.1184,
        "mean_compactness": 0.2776,
        "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471,
        "mean_symmetry": 0.2419,
        "mean_fractal_dimension": 0.07871,
        "radius_error": 1.095,
        "texture_error": 0.9053,
        "perimeter_error": 8.589,
        "area_error": 153.4,
        "smoothness_error": 0.006399,
        "compactness_error": 0.04904,
        "concavity_error": 0.05373,
        "concave_points_error": 0.01587,
        "symmetry_error": 0.03003,
        "fractal_dimension_error": 0.006193,
        "worst_radius": 25.38,
        "worst_texture": 17.33,
        "worst_perimeter": 184.6,
        "worst_area": 2019.0,
        "worst_smoothness": 0.1622,
        "worst_compactness": 0.6656,
        "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654,
        "worst_symmetry": 0.4601,
        "worst_fractal_dimension": 0.1189,
    }


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
