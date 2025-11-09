"""Tests for temp_logger module."""

import os
from unittest.mock import Mock, patch

from temp_logger import get_chicago_temperature, log_temperature


def test_get_chicago_temperature_success():
    """Test successful temperature fetch."""
    mock_response = Mock()
    mock_response.json.return_value = {"main": {"temp": 15.5}}
    mock_response.raise_for_status = Mock()

    with patch("temp_logger.requests.get", return_value=mock_response):
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test_key"}):
            temp = get_chicago_temperature()
            assert temp == 15.5


def test_get_chicago_temperature_api_error():
    """Test temperature fetch with API error."""
    with patch("temp_logger.requests.get", side_effect=Exception("API Error")):
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test_key"}):
            temp = get_chicago_temperature()
            assert temp is None


def test_get_chicago_temperature_no_api_key():
    """Test temperature fetch without API key."""
    with patch.dict(os.environ, {}, clear=True):
        mock_response = Mock()
        with patch("temp_logger.requests.get", return_value=mock_response):
            # Should still attempt to call API, just with None as key
            get_chicago_temperature()
            # Response behavior may vary, but function should not crash


def test_log_temperature_success(tmp_path):
    """Test successful temperature logging."""
    # Mock the temperature fetch and Path to use tmp_path
    with patch("temp_logger.get_chicago_temperature", return_value=20.5):
        with patch("temp_logger.Path") as mock_path:
            mock_path.return_value = tmp_path
            log_temperature()
            # Verify the directory creation was called
            mock_path.assert_called()


def test_log_temperature_no_temp():
    """Test logging when temperature fetch fails."""
    with patch("temp_logger.get_chicago_temperature", return_value=None):
        # Should return early without error
        log_temperature()
