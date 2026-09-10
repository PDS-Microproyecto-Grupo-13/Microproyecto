import pandas as pd

from ml_pipeline.data.validate import validate_frame


def test_validation_accepts_valid_foorilla_frame() -> None:
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "published": "2026-08-16T00:00:00Z",
                "y_min_usd": 50000.0,
                "y_max_usd": 80000.0,
                "target_source": "reportado",
            },
            {
                "id": 2,
                "published": "2026-08-17T00:00:00Z",
                "y_min_usd": 60000.0,
                "y_max_usd": 90000.0,
                "target_source": "estimado",
            },
        ]
    )
    report = validate_frame(frame)
    assert report["valid"] is True
    assert report["rows"] == 2
    assert report["target_source_counts"]["reportado"] == 1
    assert report["target_source_counts"]["estimado"] == 1
    assert report["target_statistics"]["y_min_usd"]["min"] == 50000.0


def test_validation_reports_empty_frame() -> None:
    report = validate_frame(pd.DataFrame())
    assert report["valid"] is False
    assert "dataset is empty" in report["errors"]


def test_validation_reports_missing_columns() -> None:
    frame = pd.DataFrame([{"id": 1, "published": "2026-08-16T00:00:00Z"}])
    report = validate_frame(frame)
    assert report["valid"] is False
    assert any("missing essential columns" in error for error in report["errors"])


def test_validation_fails_on_invalid_targets() -> None:
    # Non-positive and inverted targets
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "published": "2026-08-16T00:00:00Z",
                "y_min_usd": -100.0,
                "y_max_usd": 80000.0,
                "target_source": "reportado",
            },
            {
                "id": 2,
                "published": "2026-08-17T00:00:00Z",
                "y_min_usd": 120000.0,
                "y_max_usd": 80000.0,
                "target_source": "invalido",
            },
        ]
    )
    report = validate_frame(frame)
    assert report["valid"] is False
    assert any("non-positive y_min_usd" in error for error in report["errors"])
    assert any("inverted target salary ranges" in error for error in report["errors"])
    assert any("invalid target_source categories" in error for error in report["errors"])
