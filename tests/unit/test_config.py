"""Tests for configuration management."""

import pytest

from sproxxo.config import Settings

pytestmark = pytest.mark.unit


class TestSettings:
    """Test cases for Settings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()

        assert settings.app_name == "Sproxxo Fraud Detection API"
        assert settings.environment == "development"
        assert settings.debug is True
        assert settings.log_level == "INFO"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_is_production_property(self):
        """Test is_production property returns correct values."""
        # Test production environment
        settings = Settings(environment="production")
        assert settings.is_production is True
        assert settings.is_development is False

        # Test PRODUCTION (case insensitive)
        settings = Settings(environment="PRODUCTION")
        assert settings.is_production is True

        # Test non-production environment
        settings = Settings(environment="development")
        assert settings.is_production is False

    def test_is_development_property(self):
        """Test is_development property returns correct values."""
        # Test development environment
        settings = Settings(environment="development")
        assert settings.is_development is True
        assert settings.is_production is False

        # Test DEVELOPMENT (case insensitive)
        settings = Settings(environment="DEVELOPMENT")
        assert settings.is_development is True

        # Test non-development environment
        settings = Settings(environment="staging")
        assert settings.is_development is False

    def test_custom_values(self):
        """Test that custom values override defaults."""
        settings = Settings(app_name="Custom App", port=9000, debug=True)

        assert settings.app_name == "Custom App"
        assert settings.port == 9000
        assert settings.debug is True

    def test_prediction_threshold_validation(self):
        """Test prediction threshold bounds."""
        # Valid threshold
        settings = Settings(prediction_threshold=0.7)
        assert settings.prediction_threshold == 0.7

        # Edge cases
        settings = Settings(prediction_threshold=0.0)
        assert settings.prediction_threshold == 0.0

        settings = Settings(prediction_threshold=1.0)
        assert settings.prediction_threshold == 1.0
