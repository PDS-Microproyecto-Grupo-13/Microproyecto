from pathlib import Path

from ml_pipeline.common.io import sha256_file
from ml_pipeline.tracking.lineage import as_mlflow_tags


def test_hash_and_lineage_tags_degrade_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "value"
    path.write_text("stable", encoding="utf-8")
    assert len(sha256_file(path) or "") == 64
    assert sha256_file(tmp_path / "missing") is None
    assert as_mlflow_tags({"git_dirty": None, "candidate": True}) == {
        "lineage.git_dirty": "unknown",
        "lineage.candidate": "true",
    }


def test_collect_lineage_reads_data_manifest_fingerprint(tmp_path: Path) -> None:
    from ml_pipeline.common.io import write_json
    from ml_pipeline.settings import Settings
    from ml_pipeline.tracking.lineage import collect_lineage

    manifest_path = tmp_path / "artifacts/reports/data_manifest.json"
    write_json(manifest_path, {"dataset_fingerprint": "expected_sha256_hash"})
    (tmp_path / "params.yaml").write_text("data: {}\n", encoding="utf-8")
    settings = Settings.load(tmp_path)
    lineage = collect_lineage(settings)
    assert lineage["dataset_fingerprint"] == "expected_sha256_hash"
