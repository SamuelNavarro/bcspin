"""Tests for data generation utilities."""

import pytest

from sproxxo.models import TransactionFeatures
from sproxxo.utils.data_generator import (
    generate_fraudulent_transaction,
    generate_legitimate_transaction,
    generate_sample_transaction,
    generate_sample_transactions,
)

pytestmark = pytest.mark.unit


class TestGenerateSampleTransaction:
    """Test cases for generate_sample_transaction function."""

    def test_returns_transaction_features(self):
        """Test that function returns a TransactionFeatures instance."""
        transaction = generate_sample_transaction()
        assert isinstance(transaction, TransactionFeatures)

    def test_transaction_amount_range(self):
        """Test that transaction amount is within expected range."""
        transaction = generate_sample_transaction()
        assert 10 <= transaction.transaction_amount <= 1000

    def test_merchant_category_validity(self):
        """Test that merchant category is from valid options."""
        valid_categories = ["retail", "online", "gas", "restaurant", "travel", "grocery"]
        transaction = generate_sample_transaction()
        assert transaction.merchant_category in valid_categories

    def test_time_of_day_range(self):
        """Test that time of day is within valid range."""
        transaction = generate_sample_transaction()
        assert 0 <= transaction.time_of_day <= 23

    def test_location_coordinates_range(self):
        """Test that location coordinates are within expected ranges."""
        transaction = generate_sample_transaction()
        assert 30 <= transaction.location_lat <= 50
        assert -120 <= transaction.location_lon <= -70

    def test_average_spend_range(self):
        """Test that average spend is within expected range."""
        transaction = generate_sample_transaction()
        assert 20 <= transaction.average_spend <= 200

    def test_transactions_last_hour_range(self):
        """Test that transactions last hour is within valid range."""
        transaction = generate_sample_transaction()
        assert 0 <= transaction.transactions_last_hour <= 10

    def test_card_age_range(self):
        """Test that card age is within valid range."""
        transaction = generate_sample_transaction()
        assert 1 <= transaction.card_age_days <= 1000

    def test_is_foreign_transaction_type(self):
        """Test that is_foreign_transaction is boolean."""
        transaction = generate_sample_transaction()
        assert isinstance(transaction.is_foreign_transaction, bool)

    def test_merchant_risk_score_range(self):
        """Test that merchant risk score is within valid range."""
        transaction = generate_sample_transaction()
        assert 0 <= transaction.merchant_risk_score <= 1

    def test_randomness(self):
        """Test that multiple calls generate different transactions."""
        transactions = [generate_sample_transaction() for _ in range(10)]

        # Check that not all transactions are identical
        amounts = [t.transaction_amount for t in transactions]
        assert len(set(amounts)) > 1  # Should have different amounts


class TestGenerateSampleTransactions:
    """Test cases for generate_sample_transactions function."""

    def test_default_count(self):
        """Test that default generates 10 transactions."""
        transactions = generate_sample_transactions()
        assert len(transactions) == 10

    def test_custom_count(self):
        """Test that custom count is respected."""
        for n in [1, 5, 20]:
            transactions = generate_sample_transactions(n)
            assert len(transactions) == n

    def test_zero_count(self):
        """Test that zero count returns empty list."""
        transactions = generate_sample_transactions(0)
        assert transactions == []

    def test_all_are_transaction_features(self):
        """Test that all items are TransactionFeatures instances."""
        transactions = generate_sample_transactions(5)
        for transaction in transactions:
            assert isinstance(transaction, TransactionFeatures)

    def test_transactions_are_different(self):
        """Test that generated transactions have some variance."""
        transactions = generate_sample_transactions(5)
        amounts = [t.transaction_amount for t in transactions]
        # Very unlikely that all 5 random amounts are identical
        assert len(set(amounts)) > 1


class TestGenerateFraudulentTransaction:
    """Test cases for generate_fraudulent_transaction function."""

    def test_returns_transaction_features(self):
        """Test that function returns a TransactionFeatures instance."""
        transaction = generate_fraudulent_transaction()
        assert isinstance(transaction, TransactionFeatures)

    def test_high_transaction_amount(self):
        """Test that fraudulent transactions have high amounts."""
        transaction = generate_fraudulent_transaction()
        assert 500 <= transaction.transaction_amount <= 2000

    def test_high_risk_merchant_categories(self):
        """Test that fraudulent transactions use high-risk categories."""
        high_risk_categories = ["online", "travel"]
        transaction = generate_fraudulent_transaction()
        assert transaction.merchant_category in high_risk_categories

    def test_late_night_hours(self):
        """Test that fraudulent transactions occur during late night hours."""
        transaction = generate_fraudulent_transaction()
        assert transaction.time_of_day in [0, 1, 2, 3, 4, 5]

    def test_is_foreign_transaction(self):
        """Test that fraudulent transactions are marked as foreign."""
        transaction = generate_fraudulent_transaction()
        assert transaction.is_foreign_transaction is True

    def test_high_merchant_risk_score(self):
        """Test that fraudulent transactions have high merchant risk scores."""
        transaction = generate_fraudulent_transaction()
        assert 0.7 <= transaction.merchant_risk_score <= 1.0

    def test_many_recent_transactions(self):
        """Test that fraudulent transactions have many recent transactions."""
        transaction = generate_fraudulent_transaction()
        assert 5 <= transaction.transactions_last_hour <= 15

    def test_new_card(self):
        """Test that fraudulent transactions often use new cards."""
        transaction = generate_fraudulent_transaction()
        assert 1 <= transaction.card_age_days <= 30


class TestGenerateLegitimateTransaction:
    """Test cases for generate_legitimate_transaction function."""

    def test_returns_transaction_features(self):
        """Test that function returns a TransactionFeatures instance."""
        transaction = generate_legitimate_transaction()
        assert isinstance(transaction, TransactionFeatures)

    def test_normal_transaction_amount(self):
        """Test that legitimate transactions have normal amounts."""
        transaction = generate_legitimate_transaction()
        assert 10 <= transaction.transaction_amount <= 100

    def test_common_merchant_categories(self):
        """Test that legitimate transactions use common categories."""
        common_categories = ["grocery", "retail", "gas"]
        transaction = generate_legitimate_transaction()
        assert transaction.merchant_category in common_categories

    def test_normal_business_hours(self):
        """Test that legitimate transactions occur during normal hours."""
        transaction = generate_legitimate_transaction()
        assert 8 <= transaction.time_of_day <= 20

    def test_us_location_range(self):
        """Test that legitimate transactions are in US coordinate range."""
        transaction = generate_legitimate_transaction()
        assert 30 <= transaction.location_lat <= 50
        assert -120 <= transaction.location_lon <= -70

    def test_is_domestic_transaction(self):
        """Test that legitimate transactions are domestic."""
        transaction = generate_legitimate_transaction()
        assert transaction.is_foreign_transaction is False

    def test_low_merchant_risk_score(self):
        """Test that legitimate transactions have low merchant risk scores."""
        transaction = generate_legitimate_transaction()
        assert 0 <= transaction.merchant_risk_score <= 0.3

    def test_few_recent_transactions(self):
        """Test that legitimate transactions have few recent transactions."""
        transaction = generate_legitimate_transaction()
        assert 0 <= transaction.transactions_last_hour <= 3

    def test_established_card(self):
        """Test that legitimate transactions use established cards."""
        transaction = generate_legitimate_transaction()
        assert 100 <= transaction.card_age_days <= 1000

    def test_normal_average_spend(self):
        """Test that legitimate transactions have normal average spend."""
        transaction = generate_legitimate_transaction()
        assert 20 <= transaction.average_spend <= 100


class TestFraudulentVsLegitimate:
    """Test cases comparing fraudulent vs legitimate transaction characteristics."""

    def test_fraudulent_higher_amounts(self):
        """Test that fraudulent transactions generally have higher amounts."""
        fraudulent = generate_fraudulent_transaction()
        legitimate = generate_legitimate_transaction()

        # Fraudulent should have minimum 500, legitimate maximum 100
        assert fraudulent.transaction_amount > legitimate.transaction_amount

    def test_fraudulent_higher_risk_scores(self):
        """Test that fraudulent transactions have higher risk scores."""
        fraudulent = generate_fraudulent_transaction()
        legitimate = generate_legitimate_transaction()

        # Fraudulent minimum 0.7, legitimate maximum 0.3
        assert fraudulent.merchant_risk_score > legitimate.merchant_risk_score

    def test_foreign_vs_domestic(self):
        """Test foreign transaction flag differences."""
        fraudulent = generate_fraudulent_transaction()
        legitimate = generate_legitimate_transaction()

        assert fraudulent.is_foreign_transaction is True
        assert legitimate.is_foreign_transaction is False

    def test_card_age_differences(self):
        """Test that fraudulent transactions use newer cards."""
        fraudulent = generate_fraudulent_transaction()
        legitimate = generate_legitimate_transaction()

        # Fraudulent max 30 days, legitimate min 100 days
        assert fraudulent.card_age_days < legitimate.card_age_days
