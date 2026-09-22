from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.schemas.analytics import AnalyticsStorageEnvelope, AnalyticsSummary
from app.services.analytics_service import AnalyticsStorageService, calculate_artifact_hash

PUBLISH_TOKEN = "test-analytics-token"
AUTH_HEADERS = {"Authorization": f"Bearer {PUBLISH_TOKEN}"}


def analytics_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset": {
            "source": "fixture",
            "target_scope": "reportado",
            "counts": {
                "raw_snapshot_rows": 3,
                "validated_rows": 2,
                "modelable_rows": 2,
            },
            "data_range": {
                "published_min": "2026-01-01T00:00:00Z",
                "published_max": "2026-01-02T00:00:00Z",
            },
            "salary_midpoint": {
                "currency": "USD",
                "period": "annual",
                "mean_usd": 100000.0,
                "median_usd": 100000.0,
                "minimum_usd": 50000.0,
                "maximum_usd": 150000.0,
            },
            "salary_midpoint_distribution": [
                {
                    "lower_bound_usd": 0,
                    "upper_bound_usd": 100000,
                    "count": 1,
                    "proportion": 0.5,
                },
                {
                    "lower_bound_usd": 100000,
                    "upper_bound_usd": None,
                    "count": 1,
                    "proportion": 0.5,
                },
            ],
            "seniority_distribution": [
                {"category": "SE", "count": 1, "proportion": 0.5},
                {"category": "desconocido", "count": 1, "proportion": 0.5},
            ],
            "work_mode_distribution": [{"category": "presencial", "count": 2, "proportion": 1.0}],
            "top_technologies": [
                {"technology": "python", "count": 2, "proportion": 1.0},
                {"technology": "sql", "count": 1, "proportion": 0.5},
            ],
        },
        "model": {
            "algorithm": "lightgbm",
            "evaluation_rows": 1,
            "mae_average_usd": 25000.0,
            "r2_salary_min": 0.5,
            "r2_salary_max": 0.6,
            "predicted_range_coverage": 0.2,
            "uncertainty_margin_usd": 40000.0,
            "uncertainty_nominal_coverage": 0.8,
            "uncertainty_test_coverage": 0.75,
        },
        "metadata": {
            "dataset_fingerprint": "a" * 64,
            "snapshot_count": 1,
            "git_commit": "b" * 40,
            "dvc_yaml_hash": "c" * 64,
            "params_hash": "d" * 64,
        },
    }


def changed_payload() -> dict[str, object]:
    payload = analytics_payload()
    payload["model"]["mae_average_usd"] = 26000.0  # type: ignore[index]
    return payload


async def build_client(settings: Settings) -> AsyncClient:
    app = create_app(settings=settings)

    async def override_settings() -> Settings:
        return settings

    app.dependency_overrides[get_settings] = override_settings
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
async def test_initial_publish_then_get_exact_summary(client) -> None:
    payload = analytics_payload()
    response = await client.post("/api/v1/analytics/snapshots", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["status"] == "created"
    assert body["schema_version"] == "1.0"
    expected_hash = calculate_artifact_hash(AnalyticsSummary.model_validate(payload))
    assert body["artifact_hash"] == expected_hash

    get_response = await client.get("/api/v1/analytics/summary")
    assert get_response.status_code == 200
    assert get_response.headers["cache-control"] == "no-store"
    assert get_response.json() == payload


@pytest.mark.asyncio
async def test_identical_publish_is_unchanged_without_rewrite(client, test_settings) -> None:
    first = await client.post(
        "/api/v1/analytics/snapshots", json=analytics_payload(), headers=AUTH_HEADERS
    )
    storage_path = Path(test_settings.ANALYTICS_STORAGE_PATH)
    first_bytes = storage_path.read_bytes()
    first_stat = storage_path.stat()

    second = await client.post(
        "/api/v1/analytics/snapshots", json=analytics_payload(), headers=AUTH_HEADERS
    )

    assert second.status_code == 200
    assert second.json()["status"] == "unchanged"
    assert second.json()["stored_at"] == first.json()["stored_at"]
    assert storage_path.read_bytes() == first_bytes
    assert storage_path.stat().st_mtime_ns == first_stat.st_mtime_ns


@pytest.mark.asyncio
async def test_different_publish_replaces_snapshot(client) -> None:
    first = await client.post(
        "/api/v1/analytics/snapshots", json=analytics_payload(), headers=AUTH_HEADERS
    )
    second = await client.post(
        "/api/v1/analytics/snapshots", json=changed_payload(), headers=AUTH_HEADERS
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["status"] == "replaced"
    assert second.json()["artifact_hash"] != first.json()["artifact_hash"]
    get_response = await client.get("/api/v1/analytics/summary")
    assert get_response.json() == changed_payload()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token"}])
async def test_publish_requires_correct_token(client, headers) -> None:
    response = await client.post(
        "/api/v1/analytics/snapshots", json=analytics_payload(), headers=headers
    )

    assert response.status_code == 401
    assert response.json()["error"] == "analytics_unauthorized"


@pytest.mark.asyncio
async def test_unconfigured_publish_token_returns_503(tmp_path: Path) -> None:
    settings = Settings(
        ENVIRONMENT="test",
        ANALYTICS_PUBLISH_TOKEN=None,
        ANALYTICS_STORAGE_PATH=tmp_path / "analytics.json",
    )
    async with await build_client(settings) as isolated_client:
        response = await isolated_client.post(
            "/api/v1/analytics/snapshots", json=analytics_payload()
        )

    assert response.status_code == 503
    assert response.json()["error"] == "analytics_publish_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body.update({"generated_at": "2026-01-03T00:00:00Z"}),
        lambda body: body.update({"schema_version": "2.0"}),
        lambda body: body["metadata"].update({"dvc_revision": "e" * 64}),
        lambda body: body["dataset"]["counts"].update({"modelable_rows": "2"}),
    ],
)
async def test_publish_rejects_extra_version_and_wrong_types(client, mutation) -> None:
    payload = analytics_payload()
    mutation(payload)

    response = await client.post("/api/v1/analytics/snapshots", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


@pytest.mark.asyncio
async def test_publish_rejects_cross_field_invariant(client) -> None:
    payload = analytics_payload()
    payload["dataset"]["salary_midpoint_distribution"][0]["count"] = 0  # type: ignore[index]

    response = await client.post("/api/v1/analytics/snapshots", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


@pytest.mark.asyncio
async def test_empty_get_returns_expected_404(client) -> None:
    response = await client.get("/api/v1/analytics/summary")

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["error"] == "analytics_not_published"


@pytest.mark.asyncio
async def test_corrupt_snapshot_returns_503(client, test_settings) -> None:
    storage_path = Path(test_settings.ANALYTICS_STORAGE_PATH)
    storage_path.write_text("not-json", encoding="utf-8")

    response = await client.get("/api/v1/analytics/summary")

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["error"] == "analytics_storage_unavailable"
    assert str(storage_path) not in response.text


@pytest.mark.asyncio
async def test_storage_failure_preserves_previous_snapshot(
    client, test_settings, monkeypatch
) -> None:
    initial = analytics_payload()
    created = await client.post("/api/v1/analytics/snapshots", json=initial, headers=AUTH_HEADERS)
    assert created.status_code == 201
    storage_path = Path(test_settings.ANALYTICS_STORAGE_PATH)
    original_bytes = storage_path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure with internal path")

    monkeypatch.setattr("app.services.analytics_service.os.replace", fail_replace)
    failed = await client.post(
        "/api/v1/analytics/snapshots", json=changed_payload(), headers=AUTH_HEADERS
    )

    assert failed.status_code == 503
    assert failed.json()["error"] == "analytics_storage_unavailable"
    assert str(storage_path) not in failed.text
    assert storage_path.read_bytes() == original_bytes


def test_new_service_instance_reads_persisted_snapshot(tmp_path: Path) -> None:
    storage_path = tmp_path / "analytics.json"
    summary = AnalyticsSummary.model_validate(analytics_payload())

    first_service = AnalyticsStorageService(storage_path)
    created = first_service.publish(summary)
    second_service = AnalyticsStorageService(storage_path)

    assert created.status == "created"
    assert second_service.get_summary() == summary
    envelope = AnalyticsStorageEnvelope.model_validate_json(storage_path.read_text())
    assert envelope.summary == summary
    assert envelope.artifact_hash == calculate_artifact_hash(summary)


@pytest.mark.asyncio
async def test_token_is_not_exposed_in_logs_or_response(client, caplog) -> None:
    secret = "highly-sensitive-wrong-token"
    caplog.set_level(logging.DEBUG)

    response = await client.post(
        "/api/v1/analytics/snapshots",
        json=analytics_payload(),
        headers={"Authorization": f"Bearer {secret}"},
    )

    assert response.status_code == 401
    assert secret not in response.text
    assert secret not in caplog.text


def test_canonical_hash_is_independent_of_input_key_order() -> None:
    payload = analytics_payload()
    reversed_payload = dict(reversed(list(copy.deepcopy(payload).items())))

    first = AnalyticsSummary.model_validate(payload)
    second = AnalyticsSummary.model_validate(reversed_payload)

    assert calculate_artifact_hash(first) == calculate_artifact_hash(second)
    pretty_hash_input = json.dumps(payload, indent=2, sort_keys=True).encode()
    assert calculate_artifact_hash(first) != hashlib.sha256(pretty_hash_input).hexdigest()
