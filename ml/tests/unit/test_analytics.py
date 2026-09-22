from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from ml_pipeline.analytics import analytics, write_summary_atomic
from ml_pipeline.analytics_schema import AnalyticsSummary
from ml_pipeline.features import SKILLS
from ml_pipeline.settings import Settings
from pydantic import ValidationError

FINGERPRINT = "a" * 64
DVC_HASH = "b" * 64
PARAMS_HASH = "c" * 64
GIT_COMMIT = "d" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(
    row_id: int,
    published: str,
    salary_min: float,
    salary_max: float,
    seniority: str | None,
    work_mode: str,
    skills: set[str],
) -> dict:
    row = {
        "id": row_id,
        "published": pd.Timestamp(published),
        "y_min_usd": salary_min,
        "y_max_usd": salary_max,
        "experience_level": seniority,
        "work_mode": work_mode,
    }
    for skill in SKILLS:
        row[f"skill_{skill}"] = int(skill in skills)
    return row


@pytest.fixture
def analytics_root(tmp_path: Path) -> Path:
    (tmp_path / "params.yaml").write_text(
        """data:
  target_scope: reportado
analytics:
  salary_bins_usd: [0, 50000, 100000, 250000]
  top_technologies_limit: 2
""",
        encoding="utf-8",
    )
    frames = {
        "train": pd.DataFrame(
            [
                _row(1, "2026-01-01T00:00:00Z", 20000, 30000, "SE", "remoto", {"python", "sql"}),
                _row(2, "2026-02-01T00:00:00Z", 40000, 60000, None, "presencial", {"python"}),
            ]
        ),
        "validation": pd.DataFrame(
            [_row(3, "2026-03-01T00:00:00Z", 90000, 110000, "MI", "remoto", {"sql", "aws"})]
        ),
        "test": pd.DataFrame(
            [_row(4, "2026-04-01T00:00:00Z", 250000, 350000, "", "híbrido", {"docker"})]
        ),
    }
    for name, frame in frames.items():
        path = tmp_path / f"data/processed/{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    reports = tmp_path / "artifacts/reports"
    _write_json(
        reports / "data_manifest.json",
        {
            "source": "fixture",
            "snapshot_count": 2,
            "dataset_fingerprint": FINGERPRINT,
            "snapshots": [{"rows_raw": 10}, {"rows_raw": 5}],
        },
    )
    _write_json(reports / "validation.json", {"valid": True, "rows": 8})
    _write_json(
        reports / "preprocess.json",
        {
            "target_scope": "reportado",
            "rows_modeling": 4,
            "dataset_fingerprint": FINGERPRINT,
            "split": {
                "train": {"rows": 2},
                "validation": {"rows": 1},
                "test": {"rows": 1},
            },
        },
    )
    _write_json(
        reports / "metrics.json",
        {
            "mae_promedio": 1234.567,
            "r2_min": 0.4,
            "r2_max": 0.5,
            "cobertura_intervalo": 0.25,
            "uncertainty_margin": 9876.543,
            "uncertainty_nominal_coverage": 0.8,
            "uncertainty_test_coverage": 0.75,
        },
    )
    _write_json(
        reports / "experiment_manifest.json",
        {
            "algorithm": "lightgbm",
            "dataset_fingerprint": FINGERPRINT,
            "lineage": {
                "dataset_fingerprint": FINGERPRINT,
                "git_commit": GIT_COMMIT,
                "dvc_revision": DVC_HASH,
                "params_hash": PARAMS_HASH,
            },
        },
    )
    return tmp_path


def test_analytics_generates_expected_contract_and_is_byte_deterministic(
    analytics_root: Path,
) -> None:
    settings = Settings.load(analytics_root)
    first = analytics(settings)
    output = analytics_root / "artifacts/reports/dashboard_summary.json"
    first_bytes = output.read_bytes()
    second = analytics(settings)

    assert output.read_bytes() == first_bytes
    assert first == second
    assert first.schema_version == "1.0"
    assert first.dataset.counts.model_dump() == {
        "raw_snapshot_rows": 15,
        "validated_rows": 8,
        "modelable_rows": 4,
    }
    assert first.dataset.data_range.published_min == "2026-01-01T00:00:00Z"
    assert first.dataset.data_range.published_max == "2026-04-01T00:00:00Z"
    assert first.dataset.salary_midpoint.model_dump() == {
        "currency": "USD",
        "period": "annual",
        "mean_usd": 118750.0,
        "median_usd": 75000.0,
        "minimum_usd": 25000.0,
        "maximum_usd": 300000.0,
    }
    assert [item.count for item in first.dataset.salary_midpoint_distribution] == [1, 1, 1, 1]
    assert [(item.category, item.count) for item in first.dataset.seniority_distribution] == [
        ("desconocido", 2),
        ("MI", 1),
        ("SE", 1),
    ]
    assert [(item.technology, item.count) for item in first.dataset.top_technologies] == [
        ("python", 2),
        ("sql", 2),
    ]
    assert first.metadata.dvc_yaml_hash == DVC_HASH
    assert "dvc_revision" not in first.metadata.model_dump()
    assert first.model.evaluation_rows == 1


def test_analytics_rejects_split_count_and_fingerprint_mismatches(analytics_root: Path) -> None:
    preprocess_path = analytics_root / "artifacts/reports/preprocess.json"
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    preprocess["split"]["test"]["rows"] = 2
    _write_json(preprocess_path, preprocess)
    with pytest.raises(ValueError, match="test split row count mismatch"):
        analytics(Settings.load(analytics_root))

    preprocess["split"]["test"]["rows"] = 1
    preprocess["dataset_fingerprint"] = "e" * 64
    _write_json(preprocess_path, preprocess)
    with pytest.raises(ValueError, match="fingerprints are inconsistent"):
        analytics(Settings.load(analytics_root))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"extra": True}),
        lambda value: value.update({"schema_version": "2.0"}),
        lambda value: value["model"].update({"mae_average_usd": "123"}),
        lambda value: value["model"].update({"r2_salary_min": float("nan")}),
        lambda value: value["model"].update({"r2_salary_max": float("inf")}),
        lambda value: value["dataset"]["data_range"].update(
            {"published_min": "2027-01-01T00:00:00Z"}
        ),
        lambda value: value["dataset"]["salary_midpoint_distribution"][0].update(
            {"lower_bound_usd": 1}
        ),
        lambda value: value["dataset"]["seniority_distribution"][0].update(
            {"proportion": 0.1}
        ),
        lambda value: value["dataset"].update(
            {"top_technologies": [value["dataset"]["top_technologies"][0]] * 2}
        ),
    ],
)
def test_schema_rejects_invalid_payloads(analytics_root: Path, mutation) -> None:
    summary = analytics(Settings.load(analytics_root))
    payload = copy.deepcopy(summary.model_dump(mode="json"))
    mutation(payload)
    with pytest.raises(ValidationError):
        AnalyticsSummary.model_validate(payload)


def test_atomic_write_preserves_previous_output_on_replace_failure(
    analytics_root: Path, monkeypatch
) -> None:
    summary = analytics(Settings.load(analytics_root))
    output = analytics_root / "artifacts/reports/dashboard_summary.json"
    output.write_bytes(b"previous-valid-content\n")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("ml_pipeline.analytics.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_summary_atomic(output, summary)
    assert output.read_bytes() == b"previous-valid-content\n"
    assert not list(output.parent.glob(f".{output.name}.*"))
