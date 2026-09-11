import sys

import pytest
from mlflow.exceptions import MlflowException

from model_provider.inference.start import (
    ServingConfig,
    build_serve_command,
    load_config_from_env,
    resolve_model_info,
    wait_for_tracking_server,
)


def test_load_config_from_env_valid(monkeypatch):
    """Test loading configuration from environment variables successfully."""
    monkeypatch.setenv("MODEL_NAME", "salary_predict_model")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    monkeypatch.setenv("MODEL_ALIAS", "champion")
    monkeypatch.setenv("INFERENCE_HOST", "0.0.0.0")
    monkeypatch.setenv("INFERENCE_PORT", "5001")

    config = load_config_from_env()
    assert config.model_name == "salary_predict_model"
    assert config.tracking_uri == "http://mlflow-server:5000"
    assert config.model_alias == "champion"
    assert config.host == "0.0.0.0"
    assert config.port == 5001


def test_load_config_from_env_missing_model_name(monkeypatch):
    """Test that missing MODEL_NAME exits with code 1."""
    monkeypatch.delenv("MODEL_NAME", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        load_config_from_env()

    assert exc_info.value.code == 1


def test_load_config_from_env_invalid_port(monkeypatch):
    """Test that non-integer port exits with code 1."""
    monkeypatch.setenv("MODEL_NAME", "salary_predict_model")
    monkeypatch.setenv("INFERENCE_PORT", "invalid_port")

    with pytest.raises(SystemExit) as exc_info:
        load_config_from_env()

    assert exc_info.value.code == 1


def test_wait_for_tracking_server_success(mock_mlflow_client):
    """Test tracking server connectivity verification succeeds."""
    mock_mlflow_client.search_experiments.return_value = []
    # Should not raise
    wait_for_tracking_server(mock_mlflow_client, max_retries=2, delay=0.01)


def test_wait_for_tracking_server_failure(mock_mlflow_client):
    """Test tracking server connection failure after retries raises ConnectionError."""
    mock_mlflow_client.search_experiments.side_effect = Exception("Connection refused")
    mock_mlflow_client.tracking_uri = "http://bad-uri:5000"

    with pytest.raises(ConnectionError) as exc_info:
        wait_for_tracking_server(mock_mlflow_client, max_retries=2, delay=0.01)

    assert "Unable to connect to MLflow Tracking Server" in str(exc_info.value)


def test_resolve_model_info_success(mock_mlflow_client, mock_registered_model, mock_model_version):
    """Test resolving alias into concrete model version metadata."""
    mock_mlflow_client.get_registered_model.return_value = mock_registered_model
    mock_mlflow_client.get_model_version_by_alias.return_value = mock_model_version

    config = ServingConfig(
        tracking_uri="http://mlflow:5000",
        model_name="salary_predict_model",
        model_alias="champion",
        host="0.0.0.0",
        port=5001,
    )

    info = resolve_model_info(mock_mlflow_client, config)

    assert info.model_name == "salary_predict_model"
    assert info.model_alias == "champion"
    assert info.version == "1"
    assert info.run_id == "run-1234-abcd"


def test_resolve_model_info_missing_registered_model(mock_mlflow_client):
    """Test error when registered model does not exist."""
    mock_mlflow_client.get_registered_model.side_effect = MlflowException("Model not found")

    config = ServingConfig(
        tracking_uri="http://mlflow:5000",
        model_name="non_existent_model",
        model_alias="champion",
        host="0.0.0.0",
        port=5001,
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_model_info(mock_mlflow_client, config)

    assert "Registered model 'non_existent_model' does not exist" in str(exc_info.value)


def test_resolve_model_info_missing_alias(mock_mlflow_client, mock_registered_model):
    """Test error when alias is not found on existing registered model."""
    mock_mlflow_client.get_registered_model.return_value = mock_registered_model
    mock_mlflow_client.get_model_version_by_alias.side_effect = MlflowException("Alias not found")

    config = ServingConfig(
        tracking_uri="http://mlflow:5000",
        model_name="salary_predict_model",
        model_alias="staging",
        host="0.0.0.0",
        port=5001,
    )

    with pytest.raises(ValueError) as exc_info:
        resolve_model_info(mock_mlflow_client, config)

    assert "Alias 'staging' is not configured for model 'salary_predict_model'" in str(
        exc_info.value
    )


def test_build_serve_command():
    """Test construction of mlflow models serve CLI command."""
    config = ServingConfig(
        tracking_uri="http://mlflow:5000",
        model_name="salary_predict_model",
        model_alias="champion",
        host="0.0.0.0",
        port=5001,
        env_manager="local",
    )

    from model_provider.inference.start import ResolvedModelInfo

    model_info = ResolvedModelInfo(
        model_name="salary_predict_model",
        model_alias="champion",
        version="3",
        run_id="run-999",
        source="s3://path",
        creation_timestamp=100,
    )

    cmd = build_serve_command(config, model_info)

    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-m", "mlflow", "models", "serve"]
    assert "--model-uri" in cmd
    model_uri_idx = cmd.index("--model-uri") + 1
    assert cmd[model_uri_idx] == "models:/salary_predict_model/3"
    assert "--host" in cmd
    assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "5001"
    assert "--env-manager" in cmd
    assert cmd[cmd.index("--env-manager") + 1] == "local"


def test_load_config_from_env_salary_predictor(monkeypatch):
    """Test loading configuration specifically for canonical salary-predictor."""
    monkeypatch.setenv("MODEL_NAME", "salary-predictor")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking:5000")
    monkeypatch.setenv("MODEL_ALIAS", "champion")
    monkeypatch.setenv("INFERENCE_PORT", "5001")

    config = load_config_from_env()
    assert config.model_name == "salary-predictor"
    assert config.model_alias == "champion"
    assert config.port == 5001
    assert config.tracking_uri == "http://mlflow-tracking:5000"


def test_resolve_model_info_salary_predictor_champion(mock_mlflow_client, mock_model_version):
    """Test resolving salary-predictor@champion returns concrete version 1."""
    mock_mlflow_client.get_registered_model.return_value = {"name": "salary-predictor"}
    mock_model_version.version = "1"
    mock_model_version.run_id = "run-db1fd1"
    mock_mlflow_client.get_model_version_by_alias.return_value = mock_model_version

    config = ServingConfig(
        tracking_uri="http://localhost:5000",
        model_name="salary-predictor",
        model_alias="champion",
        host="0.0.0.0",
        port=5001,
    )

    info = resolve_model_info(mock_mlflow_client, config)
    assert info.model_name == "salary-predictor"
    assert info.model_alias == "champion"
    assert info.version == "1"
    assert info.run_id == "run-db1fd1"


def test_build_serve_command_exact_version_pinning_no_hot_reload():
    """Test that build_serve_command fixes the version immutably preventing hot-reload."""
    from model_provider.inference.start import ResolvedModelInfo

    config = ServingConfig(
        tracking_uri="http://localhost:5000",
        model_name="salary-predictor",
        model_alias="champion",
        host="0.0.0.0",
        port=5001,
    )

    model_info = ResolvedModelInfo(
        model_name="salary-predictor",
        model_alias="champion",
        version="1",
        run_id="run-db1fd1",
        source=None,
        creation_timestamp=None,
    )

    cmd = build_serve_command(config, model_info)

    # Must be pinned to concrete version /1, never to alias @champion
    assert "--model-uri" in cmd
    uri_idx = cmd.index("--model-uri") + 1
    assert cmd[uri_idx] == "models:/salary-predictor/1"
    assert "@" not in cmd[uri_idx]


def test_load_config_from_env_custom_status_port(monkeypatch):
    """Test parsing custom INFERENCE_STATUS_PORT."""
    monkeypatch.setenv("MODEL_NAME", "salary-predictor")
    monkeypatch.setenv("INFERENCE_STATUS_PORT", "5005")
    config = load_config_from_env()
    assert config.status_port == 5005


def test_load_config_from_env_invalid_status_port(monkeypatch):
    """Test invalid status port falls back to default 5002."""
    monkeypatch.setenv("MODEL_NAME", "salary-predictor")
    monkeypatch.setenv("INFERENCE_STATUS_PORT", "not_a_number")
    config = load_config_from_env()
    assert config.status_port == 5002


def test_status_server_endpoints():
    """Test InferenceStatusServer handles /status, /health, and error states correctly."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from model_provider.inference.start import InferenceStatusServer

    state = {"running": True}

    def dummy_status():
        return {
            "status": "ok" if state["running"] else "error",
            "model_name": "salary-predictor",
            "requested_alias": "champion",
            "resolved_version": "1",
            "loaded_version": "1",
            "model_server_running": state["running"],
        }

    server = InferenceStatusServer(("127.0.0.1", 0), dummy_status)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # 1. Test /status when running
        url_status = f"http://127.0.0.1:{port}/status"
        with urllib.request.urlopen(url_status) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"
            assert data["loaded_version"] == "1"
            assert data["model_server_running"] is True

        # 2. Test /health when running
        url_health = f"http://127.0.0.1:{port}/health"
        with urllib.request.urlopen(url_health) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "healthy"

        # 3. Test /status when not running returns 503
        state["running"] = False
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(url_status)
        assert exc_info.value.code == 503

        # 4. Test 404 for unknown route
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown")
        assert exc_info.value.code == 404

    finally:
        server.shutdown()
        server.server_close()

