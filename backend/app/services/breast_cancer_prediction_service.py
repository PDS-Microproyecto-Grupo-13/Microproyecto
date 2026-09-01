import math
from numbers import Real
from typing import Any

from app.core.exceptions import InferenceResponseError
from app.schemas.prediction import PredictionRequest
from app.services.inference_service import InferenceService

_FEATURE_MAPPING = (
    ("mean_radius", "mean radius"),
    ("mean_texture", "mean texture"),
    ("mean_perimeter", "mean perimeter"),
    ("mean_area", "mean area"),
    ("mean_smoothness", "mean smoothness"),
    ("mean_compactness", "mean compactness"),
    ("mean_concavity", "mean concavity"),
    ("mean_concave_points", "mean concave points"),
    ("mean_symmetry", "mean symmetry"),
    ("mean_fractal_dimension", "mean fractal dimension"),
    ("radius_error", "radius error"),
    ("texture_error", "texture error"),
    ("perimeter_error", "perimeter error"),
    ("area_error", "area error"),
    ("smoothness_error", "smoothness error"),
    ("compactness_error", "compactness error"),
    ("concavity_error", "concavity error"),
    ("concave_points_error", "concave points error"),
    ("symmetry_error", "symmetry error"),
    ("fractal_dimension_error", "fractal dimension error"),
    ("worst_radius", "worst radius"),
    ("worst_texture", "worst texture"),
    ("worst_perimeter", "worst perimeter"),
    ("worst_area", "worst area"),
    ("worst_smoothness", "worst smoothness"),
    ("worst_compactness", "worst compactness"),
    ("worst_concavity", "worst concavity"),
    ("worst_concave_points", "worst concave points"),
    ("worst_symmetry", "worst symmetry"),
    ("worst_fractal_dimension", "worst fractal dimension"),
)


class BreastCancerPredictionService:
    """Adapt the Breast Cancer classifier contract to the generic inference client."""

    def __init__(self, inference_service: InferenceService) -> None:
        self._inference_service = inference_service

    async def predict(self, features: PredictionRequest) -> int:
        """Build the model payload and return its validated class prediction."""
        response = await self._inference_service.invoke(self._build_payload(features))

        try:
            return self._parse_prediction(response)
        except (TypeError, ValueError) as exc:
            raise InferenceResponseError from exc

    @staticmethod
    def _build_payload(features: PredictionRequest) -> dict[str, Any]:
        return {
            "dataframe_split": {
                "columns": [mlflow_name for _, mlflow_name in _FEATURE_MAPPING],
                "data": [[getattr(features, public_name) for public_name, _ in _FEATURE_MAPPING]],
            }
        }

    @staticmethod
    def _parse_prediction(payload: dict[str, Any]) -> int:
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("Provider response must contain non-empty predictions")

        prediction = predictions[0]
        if isinstance(prediction, bool) or not isinstance(prediction, Real):
            raise TypeError("Provider prediction must be numeric")

        numeric_prediction = float(prediction)
        if not math.isfinite(numeric_prediction) or not numeric_prediction.is_integer():
            raise ValueError("Provider prediction must be a finite integer class")

        parsed_prediction = int(numeric_prediction)
        if parsed_prediction not in {0, 1}:
            raise ValueError("Provider prediction must be class 0 or 1")
        return parsed_prediction
