from pathlib import Path

from ml_pipeline.settings import Settings


def test_settings_loads_params_and_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "params.yaml").write_text("data:\n  random_state: 7\n", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "unit-model")
    settings = Settings.load(tmp_path)
    assert settings.section("data")["random_state"] == 7
    assert settings.mlflow_model_name == "unit-model"
    assert settings.path("x") == tmp_path / "x"
