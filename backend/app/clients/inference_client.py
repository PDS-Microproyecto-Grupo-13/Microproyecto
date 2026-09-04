from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends

from app.clients.base_http_client import BaseHttpClient
from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError


class InferenceClient(BaseHttpClient):
    """Adapter for the MLflow scoring protocol."""

    def __init__(self, base_url: str, timeout: float) -> None:
        super().__init__("mlflow-inference", base_url, timeout)

    async def predict(self, record: dict[str, object]) -> dict[str, float]:
        payload = {
            "dataframe_split": {
                "columns": list(record),
                "data": [[record[column] for column in record]],
            }
        }
        response = await self.post("/invocations", json_data=payload)
        body: dict[str, Any] = response.json()
        predictions = body.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response does not contain predictions",
                details={"response": body},
            )
        prediction = predictions[0]
        if not isinstance(prediction, dict):
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response has an unexpected shape",
                details={"response": body},
            )
        try:
            return {
                "salary_min_usd": float(prediction["salary_min_usd"]),
                "salary_max_usd": float(prediction["salary_max_usd"]),
                "salary_midpoint_usd": float(prediction["salary_midpoint_usd"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response is missing salary-range fields",
                details={"response": body},
            ) from exc


async def get_inference_client(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[InferenceClient, None]:
    async with InferenceClient(
        base_url=settings.INFERENCE_BASE_URL,
        timeout=settings.INFERENCE_TIMEOUT_SECONDS,
    ) as client:
        yield client
