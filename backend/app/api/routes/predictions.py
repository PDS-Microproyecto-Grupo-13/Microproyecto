from fastapi import APIRouter, Depends, status

from app.clients.inference_client import InferenceClient, get_inference_client
from app.core.config import Settings, get_settings
from app.schemas.prediction import (
    ModelDeployment,
    SalaryPredictionRequest,
    SalaryPredictionResponse,
    SalaryRange,
)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post(
    "",
    response_model=SalaryPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict a salary range",
)
async def predict_salary_range(
    request: SalaryPredictionRequest,
    settings: Settings = Depends(get_settings),
    inference: InferenceClient = Depends(get_inference_client),
) -> SalaryPredictionResponse:
    result = await inference.predict(request.to_mlflow_record())

    # Safely probe runtime status to attach loaded model version
    runtime_status = None
    if hasattr(inference, "get_runtime_status"):
        runtime_status = await inference.get_runtime_status()
    loaded_version = None
    if runtime_status:
        loaded_version = str(runtime_status.get("loaded_version") or runtime_status.get("resolved_version") or "")

    warnings = []
    if request.experience_years is None:
        warnings.append("No se informaron años de experiencia; el modelo imputó ese valor.")
    if not request.company:
        warnings.append("No se informó empresa; la precisión puede ser menor para este perfil.")
    return SalaryPredictionResponse(
        prediction=SalaryRange(
            minimum_usd=result["salary_min_usd"],
            maximum_usd=result["salary_max_usd"],
            midpoint_usd=result["salary_midpoint_usd"],
        ),
        model=ModelDeployment(
            name=settings.MODEL_NAME,
            alias=settings.MODEL_ALIAS,
            version=loaded_version or None,
        ),
        warnings=warnings,
    )


@router.get("/model", response_model=ModelDeployment, summary="Deployed model selector")
async def get_deployed_model(
    settings: Settings = Depends(get_settings),
    inference: InferenceClient = Depends(get_inference_client),
) -> ModelDeployment:
    runtime_status = None
    if hasattr(inference, "get_runtime_status"):
        runtime_status = await inference.get_runtime_status()
    loaded_version = None
    if runtime_status:
        loaded_version = str(runtime_status.get("loaded_version") or runtime_status.get("resolved_version") or "")

    return ModelDeployment(
        name=settings.MODEL_NAME,
        alias=settings.MODEL_ALIAS,
        version=loaded_version or None,
    )


@router.get("/status", summary="Runtime serving status and model alignment")
async def get_prediction_status(
    settings: Settings = Depends(get_settings),
    inference: InferenceClient = Depends(get_inference_client),
) -> dict[str, object]:
    runtime_status = None
    if hasattr(inference, "get_runtime_status"):
        runtime_status = await inference.get_runtime_status()
    return {
        "configured_model": settings.MODEL_NAME,
        "configured_alias": settings.MODEL_ALIAS,
        "runtime_inference": runtime_status or {"status": "unavailable"},
    }
