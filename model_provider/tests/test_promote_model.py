from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException
from mlops.scripts.promote_model import promote_model_version


def test_promote_model_version_success_with_previous_alias(mock_mlflow_client, mock_model_version):
    """Test successful model promotion from previous version to new version."""
    # Target version to promote
    target_version = MagicMock()
    target_version.version = "2"

    # Previous version currently on alias
    prev_version = MagicMock()
    prev_version.version = "1"

    mock_mlflow_client.get_model_version.return_value = target_version
    mock_mlflow_client.get_model_version_by_alias.return_value = prev_version

    prev_v, new_v = promote_model_version(
        client=mock_mlflow_client,
        model_name="salary_predict_model",
        version="2",
        alias="champion",
    )

    assert prev_v == "1"
    assert new_v == "2"

    mock_mlflow_client.set_registered_model_alias.assert_called_once_with(
        name="salary_predict_model",
        alias="champion",
        version="2",
    )


def test_promote_model_version_initial_promotion_no_prev_alias(mock_mlflow_client):
    """Test initial model promotion when no alias was previously assigned."""
    target_version = MagicMock()
    target_version.version = "1"

    mock_mlflow_client.get_model_version.return_value = target_version
    mock_mlflow_client.get_model_version_by_alias.side_effect = MlflowException("Alias not found")

    prev_v, new_v = promote_model_version(
        client=mock_mlflow_client,
        model_name="salary_predict_model",
        version="1",
        alias="champion",
    )

    assert prev_v is None
    assert new_v == "1"

    mock_mlflow_client.set_registered_model_alias.assert_called_once_with(
        name="salary_predict_model",
        alias="champion",
        version="1",
    )


def test_promote_model_version_target_version_not_found(mock_mlflow_client):
    """Test error when target model version does not exist in registry."""
    mock_mlflow_client.get_model_version.side_effect = MlflowException("Version 99 not found")

    with pytest.raises(ValueError) as exc_info:
        promote_model_version(
            client=mock_mlflow_client,
            model_name="salary_predict_model",
            version="99",
            alias="champion",
        )

    assert "Model version '99' for registered model 'salary_predict_model' does not exist" in str(
        exc_info.value
    )


def test_promote_model_version_set_alias_failure(mock_mlflow_client):
    """Test runtime error when set_registered_model_alias fails."""
    target_version = MagicMock()
    target_version.version = "2"

    mock_mlflow_client.get_model_version.return_value = target_version
    mock_mlflow_client.get_model_version_by_alias.side_effect = MlflowException("No prev alias")
    mock_mlflow_client.set_registered_model_alias.side_effect = MlflowException("Permission denied")

    with pytest.raises(RuntimeError) as exc_info:
        promote_model_version(
            client=mock_mlflow_client,
            model_name="salary_predict_model",
            version="2",
            alias="champion",
        )

    assert "Failed to assign alias 'champion' to version '2'" in str(exc_info.value)
