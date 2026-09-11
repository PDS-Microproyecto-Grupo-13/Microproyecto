from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Depends

import httpx

from app.clients.base_http_client import BaseHttpClient
from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError


class InferenceClient(BaseHttpClient):
    """Adapter for the MLflow scoring protocol."""

    def __init__(self, base_url: str, timeout: float, status_url: str | None = None) -> None:
        super().__init__("mlflow-inference", base_url, timeout)
        self.status_url = status_url

    async def predict(self, record: dict[str, object]) -> dict[str, float]:
        import math

        payload = {
            "dataframe_split": {
                "columns": list(record),
                "data": [[record[column] for column in record]],
            }
        }
        response = await self.post("/invocations", json_data=payload)
        body: dict[str, Any] = response.json()
        predictions = body.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response must contain exactly one prediction",
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
            min_usd = float(prediction["salary_min_usd"])
            max_usd = float(prediction["salary_max_usd"])
            mid_usd = float(prediction["salary_midpoint_usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response is missing salary-range fields",
                details={"response": body},
            ) from exc

        if not (math.isfinite(min_usd) and math.isfinite(max_usd) and math.isfinite(mid_usd)):
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response contains non-finite salary values",
                details={"response": body},
            )

        if min_usd <= 0 or max_usd <= 0 or mid_usd <= 0:
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response contains non-positive salary values",
                details={"response": body},
            )

        if min_usd > max_usd:
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response has minimum salary greater than maximum salary",
                details={"response": body},
            )

        expected_midpoint = (min_usd + max_usd) / 2.0
        if not math.isclose(mid_usd, expected_midpoint, rel_tol=1e-4, abs_tol=1e-2):
            raise ExternalServiceError(
                service_name=self.service_name,
                message="Inference response has inconsistent midpoint",
                details={"response": body},
            )

        return {
            "salary_min_usd": min_usd,
            "salary_max_usd": max_usd,
            "salary_midpoint_usd": mid_usd,
        }

    async def get_runtime_status(self) -> dict[str, Any] | None:
        """Safely probes the inference status server without blocking or breaking inference."""
        if not self.status_url:
            return None
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(self.status_url)
                if response.status_code == 200:
                    return response.json()
        except Exception:
            return None
        return None


async def get_inference_client(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[InferenceClient, None]:
    async with InferenceClient(
        base_url=settings.INFERENCE_BASE_URL,
        timeout=settings.INFERENCE_TIMEOUT_SECONDS,
        status_url=settings.INFERENCE_STATUS_URL,
    ) as client:
        yield client
