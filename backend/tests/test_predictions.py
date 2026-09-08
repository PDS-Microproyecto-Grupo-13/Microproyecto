import pytest

from app.clients.inference_client import get_inference_client


class StubInferenceClient:
    async def predict(self, record: dict[str, object]) -> dict[str, float]:
        assert record["title"] == "Data Scientist"
        assert record["tags"] == "python|sql"
        return {
            "salary_min_usd": 90000.0,
            "salary_max_usd": 130000.0,
            "salary_midpoint_usd": 110000.0,
        }


@pytest.mark.asyncio
async def test_prediction_endpoint(client) -> None:
    app = client._transport.app
    app.dependency_overrides[get_inference_client] = lambda: StubInferenceClient()
    try:
        response = await client.post(
            "/api/v1/predictions",
            json={
                "title": "Data Scientist",
                "experience_level": "SE",
                "experience_years": 6,
                "country": "Colombia",
                "is_remote": True,
                "company": "Example Corp",
                "company_is_agency": False,
                "technologies": ["python", "sql"],
                "topics": ["Data Science", "Machine Learning"],
            },
        )
    finally:
        app.dependency_overrides.pop(get_inference_client, None)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["minimum_usd"] == 90000.0
    assert body["prediction"]["maximum_usd"] == 130000.0
    assert body["model"]["alias"] == "champion"


@pytest.mark.asyncio
async def test_prediction_request_validation(client) -> None:
    response = await client.post(
        "/api/v1/predictions",
        json={
            "title": "x",
            "experience_level": "INVALID",
            "country": "Colombia",
        },
    )
    assert response.status_code == 422
