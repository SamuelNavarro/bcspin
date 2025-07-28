"""Tests for API validation integration."""

import pytest

from sproxxo.models import TransactionFeatures
from sproxxo.utils.validators import (
    get_overall_risk_score,
    get_risk_indicators,
    validate_transaction_features,
)

pytestmark = pytest.mark.unit


class TestAPIValidationIntegration:
    """Test cases for API validation integration."""

    def test_valid_transaction_validation(self):
        """Test validation with a valid transaction."""
        features = TransactionFeatures(
            transaction_amount=150.0,
            merchant_category="retail",
            time_of_day=14,
            location_lat=19.4326,
            location_lon=-99.1332,
            average_spend=100.0,
            transactions_last_hour=2,
            card_age_days=365,
            is_foreign_transaction=False,
            merchant_risk_score=0.2,
        )

        # Should have no validation errors
        validation_errors = validate_transaction_features(features)
        assert len(validation_errors) == 0

        # Should have low risk score
        risk_score = get_overall_risk_score(features)
        assert risk_score < 0.5

    def test_invalid_transaction_validation(self):
        """Test validation with an invalid transaction."""
        features = TransactionFeatures(
            transaction_amount=-50.0,  # Invalid: negative amount
            merchant_category="invalid_category",  # Invalid category
            time_of_day=25,  # Invalid: hour > 23
            location_lat=19.4326,
            location_lon=-99.1332,
            average_spend=100.0,
            transactions_last_hour=2,
            card_age_days=365,
            is_foreign_transaction=False,
            merchant_risk_score=0.2,
        )

        # Should have validation errors
        validation_errors = validate_transaction_features(features)
        assert len(validation_errors) > 0
        assert "transaction_amount" in validation_errors
        assert "merchant_category" in validation_errors
        assert "time_of_day" in validation_errors

    def test_high_risk_transaction(self):
        """Test risk assessment for high-risk transaction."""
        features = TransactionFeatures(
            transaction_amount=1500.0,  # High amount
            merchant_category="online",  # Higher risk category
            time_of_day=2,  # Late night
            location_lat=19.4326,
            location_lon=-99.1332,
            average_spend=300.0,  # High average spend
            transactions_last_hour=8,  # Many transactions
            card_age_days=15,  # New card
            is_foreign_transaction=True,  # Foreign transaction
            merchant_risk_score=0.9,  # High merchant risk
        )

        # Should have multiple risk indicators
        risk_indicators = get_risk_indicators(features)
        assert len(risk_indicators) > 0
        assert "high_amount" in risk_indicators
        assert "foreign_transaction" in risk_indicators
        assert "late_night" in risk_indicators
        assert "high_merchant_risk" in risk_indicators
        assert "multiple_transactions" in risk_indicators
        assert "new_card" in risk_indicators

        # Should have high overall risk score
        risk_score = get_overall_risk_score(features)
        assert risk_score > 0.7

    def test_low_risk_legitimate_transaction(self):
        """Test risk assessment for low-risk legitimate transaction."""
        features = TransactionFeatures(
            transaction_amount=50.0,  # Normal amount
            merchant_category="grocery",  # Low risk category
            time_of_day=14,  # Normal hours
            location_lat=19.4326,
            location_lon=-99.1332,
            average_spend=75.0,  # Normal average spend
            transactions_last_hour=1,  # Few transactions
            card_age_days=500,  # Established card
            is_foreign_transaction=False,  # Domestic transaction
            merchant_risk_score=0.1,  # Low merchant risk
        )

        # Should have low overall risk score
        risk_score = get_overall_risk_score(features)
        assert risk_score < 0.3

    def test_api_response_structure(self):
        """Test that API response includes all validation fields."""
        # This would be integration test with actual API
        # For now, just test the validators return expected types

        features = TransactionFeatures(
            transaction_amount=150.0,
            merchant_category="retail",
            time_of_day=14,
            location_lat=19.4326,
            location_lon=-99.1332,
            average_spend=100.0,
            transactions_last_hour=2,
            card_age_days=365,
            is_foreign_transaction=False,
            merchant_risk_score=0.2,
        )

        # Validate return types match API response model
        validation_errors = validate_transaction_features(features)
        risk_indicators = get_risk_indicators(features)
        risk_score = get_overall_risk_score(features)

        assert isinstance(validation_errors, dict)
        assert isinstance(risk_indicators, dict)
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 1
