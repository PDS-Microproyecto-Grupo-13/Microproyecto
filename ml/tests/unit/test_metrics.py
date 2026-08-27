import pandas as pd

from ml_pipeline.modeling.evaluate import calculate_metrics


def test_metrics_are_computed_for_binary_predictions() -> None:
    result = calculate_metrics(pd.Series([0, 1, 1, 0]), [0, 1, 0, 0])
    assert result == {"accuracy": 0.75, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}
