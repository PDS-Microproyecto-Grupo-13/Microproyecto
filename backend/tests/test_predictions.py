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
    assert dt.tzinfo is not None


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
    assert settings.MODEL_NAME == "salary-predictor"
    assert settings.MODEL_ALIAS == "champion"

