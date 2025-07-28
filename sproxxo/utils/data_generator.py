"""Data generation utilities for testing and development."""

import random

from ..models import TransactionFeatures


def generate_sample_transaction() -> TransactionFeatures:
    """Generate a sample transaction for testing.

    Returns
    -------
        Sample transaction features
    """
    return TransactionFeatures(
        transaction_amount=random.uniform(10, 1000),
        merchant_category=random.choice(["retail", "online", "gas", "restaurant", "travel", "grocery"]),
        time_of_day=random.randint(0, 23),
        location_lat=random.uniform(30, 50),
        location_lon=random.uniform(-120, -70),
        average_spend=random.uniform(20, 200),
        transactions_last_hour=random.randint(0, 10),
        card_age_days=random.randint(1, 1000),
        is_foreign_transaction=random.choice([True, False]),
        merchant_risk_score=random.uniform(0, 1),
    )


def generate_sample_transactions(n: int = 10) -> list[TransactionFeatures]:
    """Generate multiple sample transactions.

    Args:
        n: Number of transactions to generate

    Returns
    -------
        List of sample transaction features
    """
    return [generate_sample_transaction() for _ in range(n)]


def generate_fraudulent_transaction() -> TransactionFeatures:
    """Generate a sample transaction that is likely to be flagged as fraudulent.

    Returns
    -------
        Sample fraudulent transaction features
    """
    return TransactionFeatures(
        transaction_amount=random.uniform(500, 2000),  # High amount
        merchant_category=random.choice(["online", "travel"]),  # Higher risk categories
        time_of_day=random.choice([0, 1, 2, 3, 4, 5]),  # Late night
        location_lat=random.uniform(20, 60),  # Broader range
        location_lon=random.uniform(-180, 180),  # International
        average_spend=random.uniform(100, 500),  # High average
        transactions_last_hour=random.randint(5, 15),  # Many transactions
        card_age_days=random.randint(1, 30),  # New card
        is_foreign_transaction=True,  # Foreign transaction
        merchant_risk_score=random.uniform(0.7, 1.0),  # High risk merchant
    )


def generate_legitimate_transaction() -> TransactionFeatures:
    """Generate a sample transaction that is likely to be legitimate.

    Returns
    -------
        Sample legitimate transaction features
    """
    return TransactionFeatures(
        transaction_amount=random.uniform(10, 100),  # Normal amount
        merchant_category=random.choice(["grocery", "retail", "gas"]),  # Common categories
        time_of_day=random.randint(8, 20),  # Normal hours
        location_lat=random.uniform(30, 50),  # US range
        location_lon=random.uniform(-120, -70),  # US range
        average_spend=random.uniform(20, 100),  # Normal average
        transactions_last_hour=random.randint(0, 3),  # Few transactions
        card_age_days=random.randint(100, 1000),  # Established card
        is_foreign_transaction=False,  # Domestic transaction
        merchant_risk_score=random.uniform(0, 0.3),  # Low risk merchant
    )
