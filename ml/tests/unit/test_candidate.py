from ml_pipeline.modeling.candidate import candidate_decision


def test_candidate_gate_accepts_score_at_threshold() -> None:
    result = candidate_decision({"f1": 0.8}, {"primary_metric": "f1", "minimum_score": 0.8})
    assert result == {"eligible": True, "reasons": []}


def test_candidate_gate_rejects_low_score() -> None:
    result = candidate_decision({"f1": 0.79}, {"primary_metric": "f1", "minimum_score": 0.8})
    assert result["eligible"] is False
    assert result["reasons"]
