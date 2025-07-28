"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from sproxxo.models import TransactionFeatures

pytestmark = pytest.mark.unit


class TestTransactionFeatures:
    """Test cases for TransactionFeatures model."""

    def test_valid_transaction_features_creation(self):
        """Test creating a valid TransactionFeatures instance."""
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

        assert features.transaction_amount == 150.0
        assert features.merchant_category == "retail"
        assert features.time_of_day == 14
        assert features.location_lat == 19.4326
        assert features.location_lon == -99.1332
        assert features.is_foreign_transaction is False

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TransactionFeatures(
                transaction_amount=150.0,
                # Missing merchant_category and other required fields
            )

        errors = exc_info.value.errors()
        # Should have multiple missing field errors
        assert len(errors) > 1

        # Check that we have errors for missing fields
        missing_fields = [error["loc"][0] for error in errors if error["type"] == "missing"]
        assert "merchant_category" in missing_fields

    def test_type_conversion(self):
        """Test that Pydantic performs type conversion correctly."""
        features = TransactionFeatures(
            transaction_amount="150.5",  # String that should convert to float
            merchant_category="retail",
            time_of_day="14",  # String that should convert to int
            location_lat=19.4326,
            location_lon=-99.1332,
            average_spend=100.0,
            transactions_last_hour=2,
            card_age_days=365,
            is_foreign_transaction="false",  # String that should convert to bool
            merchant_risk_score=0.2,
        )

        # Verify types were converted
        assert isinstance(features.transaction_amount, float)
        assert features.transaction_amount == 150.5
        assert isinstance(features.time_of_day, int)
        assert features.time_of_day == 14
        assert isinstance(features.is_foreign_transaction, bool)
        assert features.is_foreign_transaction is False

    def test_invalid_type_conversion(self):
        """Test that invalid types raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TransactionFeatures(
                transaction_amount="not_a_number",  # Invalid float
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

        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert any(error["loc"][0] == "transaction_amount" for error in errors)

    def test_model_dict_export(self):
        """Test exporting model to dictionary."""
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

        features_dict = features.model_dump()

        # Verify all fields are present
        expected_fields = {
            "transaction_amount",
            "merchant_category",
            "time_of_day",
            "location_lat",
            "location_lon",
            "average_spend",
            "transactions_last_hour",
            "card_age_days",
            "is_foreign_transaction",
            "merchant_risk_score",
        }

        assert set(features_dict.keys()) == expected_fields
        assert features_dict["transaction_amount"] == 150.0
        assert features_dict["merchant_category"] == "retail"
