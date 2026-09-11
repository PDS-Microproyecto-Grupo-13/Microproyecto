from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from model_provider.scripts.check_alignment import (
    check_alignment,
    fetch_registry_version,
    fetch_runtime_status,
)


def test_fetch_registry_version_success():
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_mv

    version = fetch_registry_version(mock_client, "salary-predictor", "champion")
    assert version == "1"


def test_fetch_registry_version_failure():
    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.side_effect = MlflowException("Alias not found")

    with pytest.raises(RuntimeError) as exc_info:
        fetch_registry_version(mock_client, "salary-predictor", "champion")
    assert "Failed to resolve alias 'champion'" in str(exc_info.value)


@patch("urllib.request.urlopen")
def test_fetch_runtime_status_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"status": "ok", "loaded_version": "1", "model_server_running": true}'
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    data = fetch_runtime_status("http://localhost:5002/status")
    assert data["status"] == "ok"
    assert data["loaded_version"] == "1"
    assert data["model_server_running"] is True


@patch("urllib.request.urlopen")
def test_fetch_runtime_status_network_error(mock_urlopen):
    import urllib.error

    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    with pytest.raises(RuntimeError) as exc_info:
        fetch_runtime_status("http://localhost:5002/status")
    assert "Unable to connect to inference status server" in str(exc_info.value)


@patch("model_provider.scripts.check_alignment.MlflowClient")
@patch("model_provider.scripts.check_alignment.fetch_runtime_status")
def test_check_alignment_synchronized(mock_fetch_status, mock_client_cls):
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client_cls.return_value = mock_client

    mock_fetch_status.return_value = {
        "status": "ok",
        "loaded_version": "1",
        "model_server_running": True,
    }

    result = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert result["alignment_status"] == "synchronized"
    assert result["registry_version"] == "1"
    assert result["runtime_loaded_version"] == "1"


@patch("model_provider.scripts.check_alignment.MlflowClient")
@patch("model_provider.scripts.check_alignment.fetch_runtime_status")
def test_check_alignment_redeploy_required(mock_fetch_status, mock_client_cls):
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "2"  # Registry champion is v2
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client_cls.return_value = mock_client

    mock_fetch_status.return_value = {
        "status": "ok",
        "loaded_version": "1",  # Live container still serving v1
        "model_server_running": True,
    }

    result = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert result["alignment_status"] == "redeploy_required"
    assert result["registry_version"] == "2"
    assert result["runtime_loaded_version"] == "1"


@patch("model_provider.scripts.check_alignment.MlflowClient")
@patch("model_provider.scripts.check_alignment.fetch_runtime_status")
def test_check_alignment_runtime_unhealthy(mock_fetch_status, mock_client_cls):
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client_cls.return_value = mock_client

    mock_fetch_status.return_value = {
        "status": "error",
        "loaded_version": "1",
        "model_server_running": False,
    }

    result = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert result["alignment_status"] == "runtime_unhealthy"


@patch("model_provider.scripts.check_alignment.MlflowClient")
@patch("model_provider.scripts.check_alignment.fetch_runtime_status")
def test_check_alignment_rollback_pending(mock_fetch_status, mock_client_cls):
    """Test rollback pending state: Registry champion reverted to 1, but runtime still serves 2."""
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "1"  # Rolled back in Registry to v1
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client_cls.return_value = mock_client

    mock_fetch_status.return_value = {
        "status": "ok",
        "loaded_version": "2",  # Serving process still on v2 before restart
        "model_server_running": True,
    }

    result = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert result["alignment_status"] == "redeploy_required"
    assert result["registry_version"] == "1"
    assert result["runtime_loaded_version"] == "2"


@patch("model_provider.scripts.check_alignment.MlflowClient")
@patch("model_provider.scripts.check_alignment.fetch_runtime_status")
def test_check_alignment_post_redeploy_synchronized(mock_fetch_status, mock_client_cls):
    """Test synchronized state after container restart reloaded version 2."""
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "2"
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client_cls.return_value = mock_client

    mock_fetch_status.return_value = {
        "status": "ok",
        "loaded_version": "2",
        "model_server_running": True,
    }

    result = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert result["alignment_status"] == "synchronized"
    assert result["registry_version"] == "2"
    assert result["runtime_loaded_version"] == "2"


@patch("model_provider.scripts.check_alignment.MlflowClient")
@patch("model_provider.scripts.check_alignment.fetch_runtime_status")
def test_check_alignment_full_lifecycle_sequence(mock_fetch_status, mock_client_cls):
    """Demonstrate the entire promotion -> drift -> redeploy -> rollback lifecycle sequence."""
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_mv
    mock_client_cls.return_value = mock_client

    # 1. Initial state: v1 in Registry, v1 in Serving -> synchronized
    mock_mv.version = "1"
    mock_fetch_status.return_value = {"status": "ok", "loaded_version": "1", "model_server_running": True}
    res1 = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert res1["alignment_status"] == "synchronized"

    # 2. Champion promoted to v2 in Registry, container not restarted -> redeploy_required
    mock_mv.version = "2"
    mock_fetch_status.return_value = {"status": "ok", "loaded_version": "1", "model_server_running": True}
    res2 = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert res2["alignment_status"] == "redeploy_required"

    # 3. Serving container restarted -> now loaded_version is v2 -> synchronized
    mock_mv.version = "2"
    mock_fetch_status.return_value = {"status": "ok", "loaded_version": "2", "model_server_running": True}
    res3 = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert res3["alignment_status"] == "synchronized"

    # 4. Rollback: Champion moved back to v1 in Registry, container still running v2 -> redeploy_required
    mock_mv.version = "1"
    mock_fetch_status.return_value = {"status": "ok", "loaded_version": "2", "model_server_running": True}
    res4 = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert res4["alignment_status"] == "redeploy_required"

    # 5. Serving container restarted to complete rollback -> loaded_version is v1 -> synchronized
    mock_mv.version = "1"
    mock_fetch_status.return_value = {"status": "ok", "loaded_version": "1", "model_server_running": True}
    res5 = check_alignment("salary-predictor", "champion", "http://fake:5000", "http://fake:5002/status")
    assert res5["alignment_status"] == "synchronized"

