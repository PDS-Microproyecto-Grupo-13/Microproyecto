from __future__ import annotations

from typing import Any


def candidate_decision(metrics: dict[str, float], evaluation: dict[str, Any]) -> dict[str, Any]:
    metric = str(evaluation["primary_metric"])
    minimum = float(evaluation["minimum_score"])
    reasons: list[str] = []
    if metric not in metrics:
        reasons.append(f"primary metric '{metric}' is missing")
    elif metrics[metric] < minimum:
        reasons.append(f"{metric}={metrics[metric]:.6f} is below minimum_score={minimum:.6f}")
    return {"eligible": not reasons, "reasons": reasons}
