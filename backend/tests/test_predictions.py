from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.routes.predictions import get_breast_cancer_prediction_service
from app.core.config import Settings
from app.core.exceptions import InferenceUnavailableError
from app.main import create_app
from app.schemas.prediction import PredictionRequest, PredictionResponse


@pytest.mark.asyncio
async def test_prediction_endpoint_returns_public_response_schema(
    test_settings: Settings,
    breast_cancer_payload: dict[str, float],
) -> None:
    service = AsyncMock()
    service.predict.return_value = 1
    app = create_app(settings=test_settings)

    async def override_prediction_service() -> AsyncMock:
        return service

    app.dependency_overrides[get_breast_cancer_prediction_service] = override_prediction_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predictions",
            json=breast_cancer_payload,
        )

    assert response.status_code == 200
    assert PredictionResponse.model_validate(response.json()).prediction == 1
    request = service.predict.await_args.args[0]
    assert request == PredictionRequest(**breast_cancer_payload)


@pytest.mark.asyncio
async def test_prediction_endpoint_returns_503_when_inference_is_unavailable(
    test_settings: Settings,
    breast_cancer_payload: dict[str, float],
) -> None:
    service = AsyncMock()
    service.predict.side_effect = InferenceUnavailableError()
    app = create_app(settings=test_settings)

    async def override_prediction_service() -> AsyncMock:
        return service

    app.dependency_overrides[get_breast_cancer_prediction_service] = override_prediction_service

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/predictions",
            json=breast_cancer_payload,
        )

    assert response.status_code == 503
    assert response.json()["error"] == "inference_unavailable"
    assert response.json()["message"] == "Inference service is unavailable"
