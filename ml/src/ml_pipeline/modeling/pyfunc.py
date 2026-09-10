from __future__ import annotations

import logging
from typing import Any, Final

import joblib
import mlflow.pyfunc
from mlflow.models import ModelSignature, infer_signature
import numpy as np
import pandas as pd

from ml_pipeline.features import prepare_features
from ml_pipeline.modeling.train import predict_range

LOGGER = logging.getLogger(__name__)

RAW_INPUT_COLUMNS: Final[list[str]] = [
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
]

OUTPUT_COLUMNS: Final[list[str]] = [
    "salary_min_usd",
    "salary_max_usd",
    "salary_midpoint_usd",
]

__all__ = [
    "OUTPUT_COLUMNS",
    "RAW_INPUT_COLUMNS",
    "SalaryPredictorModel",
    "build_model_signature",
    "create_input_example",
    "verify_local_pyfunc_parity",
]


class SalaryPredictorModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc custom model wrapper for salary prediction.

    Execution flow:
    1. Raw DataFrame input -> prepare_features() (applies 24 canonical features).
    2. Uses model.joblib bundle (pipeline_min and pipeline_max LightGBM estimators).
    3. Transforms log1p targets via expm1, applies train_limits [floor, ceiling],
       and guarantees ordered ranges min <= max.
    4. Calculates salary_midpoint_usd = (salary_min_usd + salary_max_usd) / 2.0.
    5. Returns exact DataFrame with [salary_min_usd, salary_max_usd, salary_midpoint_usd].
    """

    def __init__(self, bundle: dict[str, Any] | None = None) -> None:
        self.bundle = bundle

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load model bundle artifact when restored from MLflow storage."""
        if "model_bundle" in context.artifacts:
            bundle_path = context.artifacts["model_bundle"]
            LOGGER.info("SalaryPredictorModel | loading bundle from %s", bundle_path)
            self.bundle = joblib.load(bundle_path)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Predict salary bounds and midpoint from raw input DataFrame.

        Parameters
        ----------
        context : PythonModelContext or None
            MLflow context containing model artifacts.
        model_input : pd.DataFrame
            DataFrame containing raw job postings with required columns.
        params : dict or None
            Optional inference parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: salary_min_usd, salary_max_usd, salary_midpoint_usd.
        """
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        if self.bundle is None:
            raise RuntimeError("Model bundle has not been loaded in PyFunc context")

        # 1. Prepare 24 canonical features from raw data
        features = prepare_features(model_input)

        # 2. Re-use existing postprocessed predict_range from train.py
        range_preds = predict_range(features, self.bundle)

        salary_min = range_preds[:, 0]
        salary_max = range_preds[:, 1]
        salary_midpoint = (salary_min + salary_max) / 2.0

        return pd.DataFrame(
            {
                "salary_min_usd": salary_min,
                "salary_max_usd": salary_max,
                "salary_midpoint_usd": salary_midpoint,
            },
            index=model_input.index,
        )


def create_input_example() -> pd.DataFrame:
    """Create a minimal, valid raw input DataFrame example for signature inference."""
    return pd.DataFrame([
        {
            "title": "Senior Machine Learning Engineer",
            "company": "Tech Innovators Inc",
            "company_is_agency": False,
            "countries": "United States",
            "regions": "Americas",
            "experience_level": "senior",
            "experience_years": 6.0,
            "has_remote": True,
            "work_mode": 2.0,
            "tags": "Python|Machine Learning|Docker|Kubernetes|AWS",
            "published": "2026-08-16T10:00:00Z",
        },
        {
            "title": "Backend Software Developer",
            "company": "Digital Services Corp",
            "company_is_agency": False,
            "countries": "Germany",
            "regions": "Europe",
            "experience_level": "mid",
            "experience_years": 3.0,
            "has_remote": False,
            "work_mode": 1.0,
            "tags": "Python|SQL|FastAPI",
            "published": "2026-08-20T14:30:00Z",
        },
    ])


def build_model_signature(
    model: SalaryPredictorModel,
    input_example: pd.DataFrame | None = None,
) -> ModelSignature:
    """Infer MLflow ModelSignature mapping raw input columns to exact output columns."""
    if input_example is None:
        input_example = create_input_example()
    output_example = model.predict(None, input_example)
    return infer_signature(input_example, output_example)


def verify_local_pyfunc_parity(
    bundle: dict[str, Any],
    raw_df: pd.DataFrame,
    pyfunc_model_or_loaded: Any,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict[str, bool]:
    """Verify strict numerical parity between local model.joblib and MLflow PyFunc model."""
    limits = bundle["train_limits"]
    floor = float(limits["floor"])
    ceiling = float(limits["ceiling"])

    # 1. Local inference
    local_features = prepare_features(raw_df)
    local_range = predict_range(local_features, bundle)
    local_min = local_range[:, 0]
    local_max = local_range[:, 1]
    local_mid = (local_min + local_max) / 2.0

    # 2. PyFunc inference
    if hasattr(pyfunc_model_or_loaded, "predict"):
        if isinstance(pyfunc_model_or_loaded, SalaryPredictorModel):
            pyfunc_out = pyfunc_model_or_loaded.predict(None, raw_df)
        else:
            pyfunc_out = pyfunc_model_or_loaded.predict(raw_df)
    else:
        raise TypeError("pyfunc_model_or_loaded must have a predict method")

    pyfunc_min = pyfunc_out["salary_min_usd"].to_numpy()
    pyfunc_max = pyfunc_out["salary_max_usd"].to_numpy()
    pyfunc_mid = pyfunc_out["salary_midpoint_usd"].to_numpy()
    pyfunc_all = pyfunc_out[OUTPUT_COLUMNS].to_numpy()

    # 3. Parity checks
    checks = {
        "mismo_min": bool(np.allclose(local_min, pyfunc_min, rtol=rtol, atol=atol)),
        "mismo_max": bool(np.allclose(local_max, pyfunc_max, rtol=rtol, atol=atol)),
        "mismo_midpoint": bool(np.allclose(local_mid, pyfunc_mid, rtol=rtol, atol=atol)),
        "midpoint_consistente": bool(
            np.allclose(pyfunc_mid, (pyfunc_min + pyfunc_max) / 2.0, rtol=rtol, atol=atol)
        ),
        "valores_finitos": bool(np.isfinite(pyfunc_all).all()),
        "valores_positivos": bool((pyfunc_all > 0).all()),
        "rangos_ordenados": bool((pyfunc_min <= pyfunc_max).all()),
        "limites_operativos": bool(
            (pyfunc_min >= floor - 1e-3).all() and (pyfunc_max <= ceiling + 1e-3).all()
        ),
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise ValueError(f"PyFunc parity verification failed for checks: {failed}")

    return checks
