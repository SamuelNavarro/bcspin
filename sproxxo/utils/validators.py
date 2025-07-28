"""Validation utilities for transaction data."""

from ..models import TransactionFeatures


def validate_transaction_features(features: TransactionFeatures) -> dict[str, list[str]]:
    """Validate transaction features and return any validation errors.

    Args:
        features: Transaction features to validate

    Returns
    -------
        Dictionary with validation errors by field
    """
    errors: dict[str, list[str]] = {}

    # Validate transaction amount
    if features.transaction_amount <= 0:
        errors.setdefault("transaction_amount", []).append("Transaction amount must be positive")

    if features.transaction_amount > 100000:
        errors.setdefault("transaction_amount", []).append("Transaction amount seems unusually high")

    # Validate merchant category
    valid_categories = ["retail", "online", "gas", "restaurant", "travel", "grocery"]
    if features.merchant_category not in valid_categories:
        errors.setdefault("merchant_category", []).append(
            f"Invalid merchant category. Must be one of: {valid_categories}"
        )

    # Validate time of day
    if not 0 <= features.time_of_day <= 23:
        errors.setdefault("time_of_day", []).append("Time of day must be between 0 and 23")

    # Validate location coordinates
    if not -90 <= features.location_lat <= 90:
        errors.setdefault("location_lat", []).append("Latitude must be between -90 and 90")

    if not -180 <= features.location_lon <= 180:
        errors.setdefault("location_lon", []).append("Longitude must be between -180 and 180")

    # Validate average spend
    if features.average_spend < 0:
        errors.setdefault("average_spend", []).append("Average spend cannot be negative")

    # Validate transactions last hour
    if features.transactions_last_hour < 0:
        errors.setdefault("transactions_last_hour", []).append("Transactions last hour cannot be negative")

    if features.transactions_last_hour > 100:
        errors.setdefault("transactions_last_hour", []).append("Transactions last hour seems unusually high")

    # Validate card age
    if features.card_age_days < 0:
        errors.setdefault("card_age_days", []).append("Card age cannot be negative")

    if features.card_age_days > 10000:
        errors.setdefault("card_age_days", []).append("Card age seems unusually high")

    # Validate merchant risk score
    if not 0 <= features.merchant_risk_score <= 1:
        errors.setdefault("merchant_risk_score", []).append("Merchant risk score must be between 0 and 1")

    return errors


def is_valid_transaction(features: TransactionFeatures) -> bool:
    """Check if transaction features are valid.

    Args:
        features: Transaction features to validate

    Returns
    -------
        True if valid, False otherwise
    """
    errors = validate_transaction_features(features)
    return len(errors) == 0


def get_risk_indicators(features: TransactionFeatures) -> dict[str, float]:
    """Get risk indicators for a transaction.

    Args:
        features: Transaction features

    Returns
    -------
        Dictionary of risk indicators and their scores
    """
    risk_indicators = {}

    # High transaction amount risk
    if features.transaction_amount > 500:
        risk_indicators["high_amount"] = min(features.transaction_amount / 1000, 1.0)

    # Foreign transaction risk
    if features.is_foreign_transaction:
        risk_indicators["foreign_transaction"] = 0.8

    # Late night transaction risk
    if features.time_of_day in [0, 1, 2, 3, 4, 5]:
        risk_indicators["late_night"] = 0.6

    # High merchant risk
    if features.merchant_risk_score > 0.7:
        risk_indicators["high_merchant_risk"] = features.merchant_risk_score

    # Multiple transactions risk
    if features.transactions_last_hour > 5:
        risk_indicators["multiple_transactions"] = min(features.transactions_last_hour / 10, 1.0)

    # New card risk
    if features.card_age_days < 30:
        risk_indicators["new_card"] = max(0, (30 - features.card_age_days) / 30)

    # High average spend risk
    if features.average_spend > 200:
        risk_indicators["high_average_spend"] = min(features.average_spend / 500, 1.0)

    return risk_indicators


def get_overall_risk_score(features: TransactionFeatures) -> float:
    """Calculate overall risk score for a transaction.

    Args:
        features: Transaction features

    Returns
    -------
        Overall risk score between 0 and 1
    """
    risk_indicators = get_risk_indicators(features)

    if not risk_indicators:
        return 0.0

    # Weighted average of risk indicators
    total_weight = 0.0
    weighted_sum = 0.0

    weights = {
        "high_amount": 0.2,
        "foreign_transaction": 0.25,
        "late_night": 0.1,
        "high_merchant_risk": 0.2,
        "multiple_transactions": 0.15,
        "new_card": 0.1,
        "high_average_spend": 0.1,
    }

    for indicator, score in risk_indicators.items():
        weight = weights.get(indicator, 0.1)
        weighted_sum += score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0
