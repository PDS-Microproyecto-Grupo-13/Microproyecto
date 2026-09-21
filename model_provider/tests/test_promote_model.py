from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

from model_provider.scripts.promote_model import promote_model_version


def test_promote_model_version_success_with_previous_alias(mock_mlflow_client):
    """Test successful model promotion from previous version to new version."""
    target_version = MagicMock()
    target_version.version = "2"
    target_version.status = "READY"
    target_version.tags = {"eligible": "true"}

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
    target_version.status = "READY"
    target_version.tags = {"eligible": "true"}

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


def test_promote_model_version_idempotent_already_assigned(mock_mlflow_client):
    """Test safe and idempotent behavior when alias already points to the same version."""
    target_version = MagicMock()
    target_version.version = "1"
    target_version.status = "READY"
    target_version.tags = {"eligible": "true"}

    prev_version = MagicMock()
    prev_version.version = "1"

    mock_mlflow_client.get_model_version.return_value = target_version
    mock_mlflow_client.get_model_version_by_alias.return_value = prev_version

    prev_v, new_v = promote_model_version(
        client=mock_mlflow_client,
        model_name="salary_predict_model",
        version="1",
        alias="champion",
    )

    assert prev_v == "1"
    assert new_v == "1"
    mock_mlflow_client.set_registered_model_alias.assert_not_called()


def test_promote_model_version_registered_model_not_found(mock_mlflow_client):
    """Test error when registered model does not exist in registry."""
    mock_mlflow_client.get_registered_model.side_effect = MlflowException(
        "Registered model not found"
    )

    with pytest.raises(ValueError) as exc_info:
        promote_model_version(
            client=mock_mlflow_client,
            model_name="nonexistent-model",
            version="1",
            alias="champion",
        )

    assert "Registered model 'nonexistent-model' does not exist" in str(exc_info.value)
    mock_mlflow_client.set_registered_model_alias.assert_not_called()


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
    mock_mlflow_client.set_registered_model_alias.assert_not_called()


def test_promote_model_version_rejects_status_not_ready(mock_mlflow_client):
    """Test error when model version status is not READY."""
    target_version = MagicMock()
    target_version.version = "1"
    target_version.status = "FAILED_REGISTRATION"
    target_version.tags = {"eligible": "true"}

    mock_mlflow_client.get_model_version.return_value = target_version

    with pytest.raises(ValueError) as exc_info:
        promote_model_version(
            client=mock_mlflow_client,
            model_name="salary_predict_model",
            version="1",
            alias="champion",
        )

    assert "Expected 'READY'" in str(exc_info.value)
    mock_mlflow_client.set_registered_model_alias.assert_not_called()


def test_promote_model_version_rejects_missing_eligible_tag(mock_mlflow_client):
    """Test error when model version has no eligible tag."""
    target_version = MagicMock()
    target_version.version = "1"
    target_version.status = "READY"
    target_version.tags = {}

    mock_mlflow_client.get_model_version.return_value = target_version

    with pytest.raises(ValueError) as exc_info:
        promote_model_version(
            client=mock_mlflow_client,
            model_name="salary_predict_model",
            version="1",
            alias="champion",
        )

    assert "because tag 'eligible' is 'None'. Expected 'true'." in str(exc_info.value)
    mock_mlflow_client.set_registered_model_alias.assert_not_called()


def test_promote_model_version_rejects_eligible_false(mock_mlflow_client):
    """Test error when model version has eligible tag set to false."""
    target_version = MagicMock()
    target_version.version = "1"
    target_version.status = "READY"
    target_version.tags = {"eligible": "false"}

    mock_mlflow_client.get_model_version.return_value = target_version

    with pytest.raises(ValueError) as exc_info:
        promote_model_version(
            client=mock_mlflow_client,
            model_name="salary_predict_model",
            version="1",
            alias="champion",
        )

    assert "because tag 'eligible' is 'false'. Expected 'true'." in str(exc_info.value)
    mock_mlflow_client.set_registered_model_alias.assert_not_called()



def test_promote_model_version_set_alias_failure(mock_mlflow_client):
    """Test runtime error when set_registered_model_alias fails."""
    target_version = MagicMock()
    target_version.version = "2"
    target_version.status = "READY"
    target_version.tags = {"eligible": "true"}

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

