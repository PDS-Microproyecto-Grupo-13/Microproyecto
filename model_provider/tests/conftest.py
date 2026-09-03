from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_mlflow_client():
    """Provides a mocked MlflowClient instance."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_model_version():
    """Provides a mocked ModelVersion instance."""
    mv = MagicMock()
    mv.version = "1"
    mv.run_id = "run-1234-abcd"
    mv.source = "file:///var/lib/mlflow/artifacts/1/models"
    mv.creation_timestamp = 1724000000000
    mv.status = "READY"
    mv.tags = {"env": "prod"}
    return mv


@pytest.fixture
def mock_registered_model(mock_model_version):
    """Provides a mocked RegisteredModel instance."""
    rm = MagicMock()
    rm.name = "salary_predict_model"
    rm.description = "Salary prediction demo model"
    rm.creation_timestamp = 1724000000000
    rm.last_updated_timestamp = 1724000500000
    rm.aliases = {"champion": "1"}
    rm.latest_versions = [mock_model_version]
    return rm
