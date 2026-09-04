from unittest.mock import MagicMock, patch

from model_provider.inference.healthcheck import check_health


def test_check_health_success():
    """Test healthcheck returns True when /health endpoint returns 200."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        assert check_health(host="127.0.0.1", port=5001) is True


def test_check_health_fallback_to_ping():
    """Test healthcheck falls back to /ping when /health fails."""
    mock_200 = MagicMock()
    mock_200.status = 200
    mock_200.__enter__.return_value = mock_200

    import urllib.error

    def side_effect(req, timeout=3.0):
        if req.full_url.endswith("/health"):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        return mock_200

    with patch("urllib.request.urlopen", side_effect=side_effect):
        assert check_health(host="127.0.0.1", port=5001) is True


def test_check_health_all_endpoints_fail():
    """Test healthcheck returns False when all endpoints fail."""
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        assert check_health(host="127.0.0.1", port=5001) is False
