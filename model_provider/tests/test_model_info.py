import pytest
from mlflow.exceptions import MlflowException
from mlops.scripts.model_info import format_timestamp, inspect_model


def test_format_timestamp():
    """Test timestamp formatter with valid epoch millis and None."""
    assert format_timestamp(None) == "N/A"
    formatted = format_timestamp(1700000000000)
    assert "2023" in formatted
    assert "UTC" in formatted


def test_inspect_model_general_success(capsys, mock_mlflow_client, mock_registered_model):
    """Test general model overview output."""
    mock_mlflow_client.get_registered_model.return_value = mock_registered_model

    inspect_model(
        client=mock_mlflow_client,
        model_name="salary_predict_model",
    )

    captured = capsys.readouterr()
    assert "Registered Model: salary_predict_model" in captured.out
    assert "Version 1" in captured.out
    assert "run-1234-abcd" in captured.out


def test_inspect_model_by_alias_success(
    capsys, mock_mlflow_client, mock_registered_model, mock_model_version
):
    """Test inspecting model resolved by alias."""
    mock_mlflow_client.get_registered_model.return_value = mock_registered_model
    mock_mlflow_client.get_model_version_by_alias.return_value = mock_model_version

    inspect_model(
        client=mock_mlflow_client,
        model_name="salary_predict_model",
        alias="champion",
    )

    captured = capsys.readouterr()
    assert "Alias 'champion' -> Version 1" in captured.out
    assert "Run ID:      run-1234-abcd" in captured.out


def test_inspect_model_by_version_success(
    capsys, mock_mlflow_client, mock_registered_model, mock_model_version
):
    """Test inspecting model by explicit version."""
    mock_mlflow_client.get_registered_model.return_value = mock_registered_model
    mock_mlflow_client.get_model_version.return_value = mock_model_version

    inspect_model(
        client=mock_mlflow_client,
        model_name="salary_predict_model",
        version="1",
    )

    captured = capsys.readouterr()
    assert "Version Details - Version 1" in captured.out
    assert "Source URI:  file:///var/lib/mlflow/artifacts/1/models" in captured.out


def test_inspect_model_not_found_exits(mock_mlflow_client):
    """Test that missing registered model triggers system exit 1."""
    mock_mlflow_client.get_registered_model.side_effect = MlflowException("Model not found")

    with pytest.raises(SystemExit) as exc_info:
        inspect_model(
            client=mock_mlflow_client,
            model_name="missing_model",
        )

    assert exc_info.value.code == 1


def test_inspect_model_alias_not_found_exits(mock_mlflow_client, mock_registered_model):
    """Test that missing alias on model triggers system exit 1."""
    mock_mlflow_client.get_registered_model.return_value = mock_registered_model
    mock_mlflow_client.get_model_version_by_alias.side_effect = MlflowException("Alias not found")

    with pytest.raises(SystemExit) as exc_info:
        inspect_model(
            client=mock_mlflow_client,
            model_name="salary_predict_model",
            alias="non_existent_alias",
        )

    assert exc_info.value.code == 1
