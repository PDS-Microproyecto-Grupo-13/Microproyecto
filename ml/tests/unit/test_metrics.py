import numpy as np
import pytest

from ml_pipeline.modeling.qualify import compute_metrics


def test_metrics_are_computed_for_salary_predictions() -> None:
    limits = {"floor": 10000.0, "ceiling": 200000.0}
    y_true = np.array([
        [50000.0, 70000.0],
        [80000.0, 100000.0],
        [120000.0, 150000.0],
    ])
    raw_pred = np.array([
        [52000.0, 68000.0],
        [78000.0, 105000.0],
        [125000.0, 145000.0],
    ])
    metrics, postprocessed = compute_metrics(y_true, raw_pred, limits)

    assert "mae_min" in metrics
    assert "mae_max" in metrics
    assert "mae_promedio" in metrics
    assert "rmse_min" in metrics
    assert "rmse_max" in metrics
    assert "mape_min" in metrics
    assert "mape_max" in metrics
    assert "r2_min" in metrics
    assert "r2_max" in metrics
    assert "mae_amplitud" in metrics
    assert "cobertura_intervalo" in metrics
    assert "incoherencia_raw" in metrics
    assert "prediccion_no_positiva" in metrics

    assert abs(metrics["mae_promedio"] - (metrics["mae_min"] + metrics["mae_max"]) / 2.0) < 1e-4
    assert metrics["prediccion_no_positiva"] == 0.0
    assert postprocessed.shape == (3, 2)

