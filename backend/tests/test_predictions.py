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


EXPECTED_PYFUNC_COLUMNS = {
    "title",
    "company",
    "company_is_agency",
    "countries",
    "regions",
    "experience_level",
    "experience_years",
    "has_remote",
    "work_mode",
    "tags",
    "published",
}


def test_to_mlflow_record_minimal_request_produces_11_columns():
    from datetime import datetime
    from app.schemas.prediction import SalaryPredictionRequest

    req = SalaryPredictionRequest(
        title="Software Engineer",
        experience_level="MI",
        country="Colombia",
    )
    record = req.to_mlflow_record()
    assert set(record.keys()) == EXPECTED_PYFUNC_COLUMNS
    assert len(record) == 11
    assert record["title"] == "Software Engineer"
    assert record["countries"] == "Colombia"
    assert record["regions"] == "desconocido"
    assert record["has_remote"] is False
    assert record["work_mode"] is None
    assert record["company"] == "Sin información"
    assert record["company_is_agency"] is False
    assert record["tags"] == ""
    assert record["experience_years"] is None
    dt = datetime.fromisoformat(str(record["published"]))
    # MLflow requiere UTC representado sin información de zona horaria.
    assert dt.tzinfo is None


def test_to_mlflow_record_region_preserved():
    from app.schemas.prediction import SalaryPredictionRequest

    req = SalaryPredictionRequest(
        title="ML Engineer",
        experience_level="SE",
        country="Germany",
        region="Europe",
    )
    record = req.to_mlflow_record()
    assert record["regions"] == "Europe"


def test_to_mlflow_record_regions_alternative_preserved():
    from app.schemas.prediction import SalaryPredictionRequest

    req = SalaryPredictionRequest(
        title="ML Engineer",
        experience_level="SE",
        country="Germany",
        regions="Europe",
    )
    record = req.to_mlflow_record()
    assert record["regions"] == "Europe"


def test_to_mlflow_record_work_mode_preserved():
    from app.schemas.prediction import SalaryPredictionRequest

    req = SalaryPredictionRequest(
        title="Data Engineer",
        experience_level="SE",
        country="Brazil",
        is_remote=True,
        work_mode=2,
    )
    record = req.to_mlflow_record()
    assert record["has_remote"] is True
    assert record["work_mode"] == 2


def test_to_mlflow_record_technologies_mapped_to_tags():
    from app.schemas.prediction import SalaryPredictionRequest

    req = SalaryPredictionRequest(
        title="DevOps Engineer",
        experience_level="SE",
        country="United States",
        technologies=["docker", "kubernetes", "aws"],
    )
    record = req.to_mlflow_record()
    assert record["tags"] == "docker|kubernetes|aws"


def test_model_name_settings():
    from app.core.config import Settings

    settings = Settings()
    assert settings.MODEL_NAME == "salary_predict_model"
    assert settings.MODEL_ALIAS == "champion"
    assert settings.INFERENCE_STATUS_URL == "http://inference:5002/status"


@pytest.mark.asyncio
async def test_prediction_endpoint_includes_runtime_version(client) -> None:
    class VersionedInferenceClient:
        async def predict(self, record: dict[str, object]) -> dict[str, float]:
            return {
                "salary_min_usd": 80000.0,
                "salary_max_usd": 120000.0,
                "salary_midpoint_usd": 100000.0,
            }

        async def get_runtime_status(self) -> dict[str, object]:
            return {
                "status": "ok",
                "loaded_version": "1",
                "model_name": "salary_predict_model",
            }

    app = client._transport.app
    app.dependency_overrides[get_inference_client] = lambda: VersionedInferenceClient()
    try:
        response = await client.post(
            "/api/v1/predictions",
            json={
                "title": "Machine Learning Engineer",
                "experience_level": "MI",
                "country": "Colombia",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"]["name"] == "salary_predict_model"
        assert body["model"]["alias"] == "champion"
        assert body["model"]["version"] == "1"

        # Also test GET /api/v1/predictions/model
        model_resp = await client.get("/api/v1/predictions/model")
        assert model_resp.status_code == 200
        model_body = model_resp.json()
        assert model_body["name"] == "salary_predict_model"
        assert model_body["version"] == "1"

        # Also test GET /api/v1/predictions/status
        status_resp = await client.get("/api/v1/predictions/status")
        assert status_resp.status_code == 200
        status_body = status_resp.json()
        assert status_body["configured_model"] == "salary_predict_model"
        assert status_body["runtime_inference"]["loaded_version"] == "1"
    finally:
        app.dependency_overrides.pop(get_inference_client, None)


