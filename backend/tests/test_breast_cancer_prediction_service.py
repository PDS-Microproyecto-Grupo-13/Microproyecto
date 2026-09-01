from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import InferenceResponseError
from app.schemas.prediction import PredictionRequest
from app.services.breast_cancer_prediction_service import BreastCancerPredictionService
from app.services.inference_service import InferenceService

EXPECTED_COLUMNS = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


@pytest.mark.asyncio
async def test_predict_builds_exact_model_payload(
    breast_cancer_payload: dict[str, float],
) -> None:
    inference_service = AsyncMock(spec=InferenceService)
    inference_service.invoke.return_value = {"predictions": [0]}
    service = BreastCancerPredictionService(inference_service)

    prediction = await service.predict(PredictionRequest(**breast_cancer_payload))

    assert prediction == 0
    inference_service.invoke.assert_awaited_once_with(
        {
            "dataframe_split": {
                "columns": EXPECTED_COLUMNS,
                "data": [list(breast_cancer_payload.values())],
            }
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_prediction", "expected"),
    [(0, 0), (1, 1), (1.0, 1)],
    ids=["class-zero", "class-one", "integral-float"],
)
async def test_predict_returns_valid_class(
    breast_cancer_payload: dict[str, float],
    provider_prediction: int | float,
    expected: int,
) -> None:
    inference_service = AsyncMock(spec=InferenceService)
    inference_service.invoke.return_value = {"predictions": [provider_prediction]}
    service = BreastCancerPredictionService(inference_service)

    prediction = await service.predict(PredictionRequest(**breast_cancer_payload))

    assert prediction == expected
    assert isinstance(prediction, int)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_response",
    [
        {"predictions": [0.42]},
        {},
        {"predictions": []},
        {"predictions": ["1"]},
        {"predictions": [float("nan")]},
        {"predictions": [float("inf")]},
        {"predictions": [True]},
        {"predictions": [2]},
    ],
    ids=[
        "non-integral",
        "missing",
        "empty",
        "non-numeric",
        "nan",
        "infinite",
        "boolean",
        "unknown-class",
    ],
)
async def test_predict_rejects_invalid_model_response(
    breast_cancer_payload: dict[str, float],
    provider_response: dict[str, object],
) -> None:
    inference_service = AsyncMock(spec=InferenceService)
    inference_service.invoke.return_value = provider_response
    service = BreastCancerPredictionService(inference_service)

    with pytest.raises(InferenceResponseError):
        await service.predict(PredictionRequest(**breast_cancer_payload))
