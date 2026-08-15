"""Configuration for the Foorilla ingestion module.

All secrets/config come from environment variables (loaded from .env in
local dev via python-dotenv). Nothing sensitive is hardcoded or committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (no-op in CI/prod where real env vars are injected)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and fill it in, or set it in your shell/CI secrets."
        )
    return value


@dataclass(frozen=True)
class FoorillaConfig:
    api_key: str = field(default_factory=lambda: _require_env("FOORILLA_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("FOORILLA_BASE_URL", "https://foorilla.com/api/v1"))
    # Foorilla's documented limits: 600 requests/minute AND 5 requests/second.
    # The per-second cap is the binding constraint (5/sec = 300/min < 600/min),
    # so that's what we throttle against.
    requests_per_second: float = float(os.getenv("FOORILLA_RATE_LIMIT_RPS", "5"))
    page_size: int = int(os.getenv("FOORILLA_PAGE_SIZE", "100"))  # API allows 1-1000
    timeout_seconds: int = int(os.getenv("FOORILLA_TIMEOUT_SECONDS", "30"))
    max_retries: int = int(os.getenv("FOORILLA_MAX_RETRIES", "5"))


def get_config() -> FoorillaConfig:
    return FoorillaConfig()
