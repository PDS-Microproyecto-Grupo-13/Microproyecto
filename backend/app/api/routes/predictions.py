from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.config import Settings, get_settings
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.breast_cancer_prediction_service import BreastCancerPredictionService
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/predictions", tags=["Predictions"])


async def get_breast_cancer_prediction_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[BreastCancerPredictionService, None]:
    """Provide the model adapter with a request-scoped inference client."""
    async with InferenceService(
        base_url=settings.INFERENCE_BASE_URL,
        timeout_seconds=settings.INFERENCE_TIMEOUT_SECONDS,
    ) as inference_service:
        yield BreastCancerPredictionService(inference_service)


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Breast Cancer Class",
    description="Returns a class prediction from the configured Breast Cancer model.",
)
async def predict_breast_cancer(
    request: PredictionRequest,
    prediction_service: Annotated[
        BreastCancerPredictionService, Depends(get_breast_cancer_prediction_service)
    ],
) -> PredictionResponse:
    """Validate public input and delegate to the model-specific service."""
    prediction = await prediction_service.predict(request)
    return PredictionResponse(prediction=prediction)
