from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from ml_pipeline.publish_analytics import PublishAnalyticsError, publish_analytics
from ml_pipeline.settings import Settings

TOKEN = "super-secret-publisher-token"
HASH = "a" * 64


def _valid_summary() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {
            "source": "fixture",
            "target_scope": "reportado",
            "counts": {"raw_snapshot_rows": 1, "validated_rows": 1, "modelable_rows": 1},
            "data_range": {
                "published_min": "2026-01-01T00:00:00Z",
                "published_max": "2026-01-01T00:00:00Z",
            },
            "salary_midpoint": {
                "currency": "USD",
                "period": "annual",
                "mean_usd": 100.0,
                "median_usd": 100.0,
                "minimum_usd": 100.0,
                "maximum_usd": 100.0,
            },
            "salary_midpoint_distribution": [
                {
                    "lower_bound_usd": 0,
                    "upper_bound_usd": None,
                    "count": 1,
                    "proportion": 1.0,
                }
            ],
            "seniority_distribution": [
                {"category": "SE", "count": 1, "proportion": 1.0}
            ],
            "work_mode_distribution": [
                {"category": "remoto", "count": 1, "proportion": 1.0}
            ],
            "top_technologies": [
                {"technology": "python", "count": 1, "proportion": 1.0}
            ],
        },
        "model": {
            "algorithm": "lightgbm",
            "evaluation_rows": 1,
            "mae_average_usd": 10.0,
            "r2_salary_min": 0.4,
            "r2_salary_max": 0.5,
            "predicted_range_coverage": 0.2,
            "uncertainty_margin_usd": 20.0,
            "uncertainty_nominal_coverage": 0.8,
            "uncertainty_test_coverage": 0.75,
        },
        "metadata": {
            "dataset_fingerprint": "b" * 64,
            "snapshot_count": 1,
            "git_commit": "c" * 40,
            "dvc_yaml_hash": "d" * 64,
            "params_hash": "e" * 64,
        },
    }


@pytest.fixture
def publish_root(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "params.yaml").write_text("data: {}\n", encoding="utf-8")
    artifact = tmp_path / "artifacts/reports/dashboard_summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(_valid_summary()), encoding="utf-8")
    monkeypatch.setenv(
        "ANALYTICS_PUBLISH_URL", "https://backend.example/api/v1/analytics/snapshots"
    )
    monkeypatch.setenv("ANALYTICS_PUBLISH_TOKEN", TOKEN)
    monkeypatch.setenv("ANALYTICS_PUBLISH_TIMEOUT_SECONDS", "3")
    return tmp_path


def _success_response(status: str) -> dict:
    return {
        "artifact_hash": HASH,
        "schema_version": "1.0",
        "status": status,
        "stored_at": "2026-09-22T18:30:00Z",
    }


@pytest.mark.parametrize(
    ("http_status", "publication_status"),
    [(201, "created"), (200, "replaced"), (200, "unchanged")],
)
def test_publish_success_sends_exact_contract_and_headers(
    publish_root: Path, http_status: int, publication_status: str
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(http_status, json=_success_response(publication_status))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = publish_analytics(Settings.load(publish_root), client=client)

    assert result.status == publication_status
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "https://backend.example/api/v1/analytics/snapshots"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == _valid_summary()


def test_missing_or_invalid_artifact_uses_exit_2_without_request(
    publish_root: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(201, json=_success_response("created"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    artifact = publish_root / "artifacts/reports/dashboard_summary.json"
    artifact.unlink()
    with pytest.raises(PublishAnalyticsError) as missing:
        publish_analytics(Settings.load(publish_root), client=client)
    assert missing.value.exit_code == 2
    assert calls == 0

    artifact.write_text('{"schema_version":"2.0"}', encoding="utf-8")
    with pytest.raises(PublishAnalyticsError) as invalid:
        publish_analytics(Settings.load(publish_root), client=client)
    assert invalid.value.exit_code == 2
    assert calls == 0


def test_missing_configuration_uses_exit_3(publish_root: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANALYTICS_PUBLISH_TOKEN")
    settings = Settings.load(publish_root)
    settings = replace(settings, analytics_publish_token=None)
    with pytest.raises(PublishAnalyticsError) as error:
        publish_analytics(settings, client=httpx.Client(transport=httpx.MockTransport(lambda r: None)))
    assert error.value.exit_code == 3

    settings = replace(
        settings,
        analytics_publish_token=TOKEN,
        analytics_publish_timeout_seconds="invalid",
    )
    with pytest.raises(PublishAnalyticsError) as invalid_timeout:
        publish_analytics(settings, client=httpx.Client(transport=httpx.MockTransport(lambda r: None)))
    assert invalid_timeout.value.exit_code == 3


@pytest.mark.parametrize(("status_code", "exit_code"), [(401, 4), (422, 4), (500, 5), (503, 5)])
def test_http_errors_map_to_exit_codes(
    publish_root: Path, status_code: int, exit_code: int
) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status_code, json={"message": "rejected"})
        )
    )
    with pytest.raises(PublishAnalyticsError) as error:
        publish_analytics(Settings.load(publish_root), client=client)
    assert error.value.exit_code == exit_code


def test_backend_error_message_cannot_echo_token(publish_root: Path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"message": f"bad token {TOKEN}"})
        )
    )
    with pytest.raises(PublishAnalyticsError) as error:
        publish_analytics(Settings.load(publish_root), client=client)
    assert TOKEN not in str(error.value)
    assert "<redacted>" in str(error.value)


@pytest.mark.parametrize(
    "transport_error",
    [httpx.ConnectError("connect failed"), httpx.ReadTimeout("timed out")],
)
def test_transport_errors_use_exit_6_and_never_expose_token(
    publish_root: Path, transport_error: Exception, caplog, capsys
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise transport_error

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(PublishAnalyticsError) as error:
        publish_analytics(Settings.load(publish_root), client=client)
    assert error.value.exit_code == 6
    captured = capsys.readouterr()
    visible = str(error.value) + caplog.text + captured.out + captured.err
    assert TOKEN not in visible


def test_invalid_success_response_uses_exit_5(publish_root: Path) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(201, json={"status": "created"}))
    )
    with pytest.raises(PublishAnalyticsError) as error:
        publish_analytics(Settings.load(publish_root), client=client)
    assert error.value.exit_code == 5


def test_cli_returns_publisher_specific_exit_code(monkeypatch) -> None:
    from ml_pipeline import cli

    def fail(settings: Settings) -> None:
        raise PublishAnalyticsError("local validation failed", 2)

    monkeypatch.setitem(cli.COMMANDS, "publish-analytics", fail)
    assert cli.main(["publish-analytics"]) == 2
