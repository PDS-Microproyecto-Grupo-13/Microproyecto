from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ml_pipeline.common.io import read_yaml


@dataclass(frozen=True)
class Settings:
    root: Path
    params: dict[str, Any]
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
    require_clean_git: bool
    analytics_publish_url: str | None = None
    analytics_publish_token: str | None = field(default=None, repr=False)
    analytics_publish_timeout_seconds: str | None = None

    @classmethod
    def load(cls, root: Path | None = None) -> "Settings":
        default_root = Path(__file__).resolve().parents[2]
        resolved_root = Path(root or os.getenv("ML_PIPELINE_ROOT", default_root)).resolve()
        load_dotenv(resolved_root / ".env", override=False)
        params_path = resolved_root / "params.yaml"
        if not params_path.is_file():
            raise FileNotFoundError(f"Configuration not found: {params_path}")
        return cls(
            root=resolved_root,
            params=read_yaml(params_path),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "salary-prediction"),
            mlflow_model_name=os.getenv("MLFLOW_MODEL_NAME", "salary_predict_model"),
            require_clean_git=os.getenv("ML_REQUIRE_CLEAN_GIT", "false").lower()
            in {"1", "true", "yes"},
            analytics_publish_url=os.getenv("ANALYTICS_PUBLISH_URL"),
            analytics_publish_token=os.getenv("ANALYTICS_PUBLISH_TOKEN"),
            analytics_publish_timeout_seconds=os.getenv(
                "ANALYTICS_PUBLISH_TIMEOUT_SECONDS", "10"
            ),
        )

    def path(self, relative: str) -> Path:
        return self.root / relative

    def section(self, name: str) -> dict[str, Any]:
        value = self.params.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"params.yaml section '{name}' must be a mapping")
        return value
