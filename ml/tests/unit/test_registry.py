from pathlib import Path

import pytest

from ml_pipeline.common.io import write_json
from ml_pipeline.settings import Settings
from ml_pipeline.tracking.registry import RegistrationError, register_candidate


def test_registration_rejects_ineligible_candidate(tmp_path: Path) -> None:
    (tmp_path / "params.yaml").write_text("data: {}\n", encoding="utf-8")
    write_json(tmp_path / "artifacts/reports/candidate.json", {"eligible": False, "reasons": ["low score"]})
    write_json(tmp_path / "artifacts/reports/tracking.json", {"run_id": "unused"})
    with pytest.raises(RegistrationError, match="not an eligible"):
        register_candidate(Settings.load(tmp_path))
