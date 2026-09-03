import pandas as pd

from ml_pipeline.data.validate import expected_columns, validate_frame


def test_validation_accepts_expected_numeric_schema() -> None:
    frame = pd.DataFrame([[0.0] * 30 + [1]], columns=expected_columns())
    assert validate_frame(frame)["valid"] is True


def test_validation_reports_missing_target_and_empty_data() -> None:
    frame = pd.DataFrame(columns=expected_columns()[:-1])
    report = validate_frame(frame)
    assert report["valid"] is False
    assert "dataset is empty" in report["errors"]
    assert any("target" in error for error in report["errors"])
