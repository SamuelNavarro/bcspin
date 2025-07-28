"""Fraud detection model implementation using XGBoost."""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..config import settings

logger = logging.getLogger(__name__)


class TransactionFeatures(BaseModel):
    """Transaction features for fraud detection."""

    transaction_amount: float = Field(..., description="Transaction amount in MXN")
    merchant_category: str = Field(..., description="Merchant category code")
    time_of_day: int = Field(..., description="Hour of day (0-23)")
    location_lat: float = Field(..., description="Transaction latitude")
    location_lon: float = Field(..., description="Transaction longitude")
    average_spend: float = Field(..., description="Average daily spend for the card")
    transactions_last_hour: int = Field(..., description="Number of transactions in the last hour")
    card_age_days: int = Field(..., description="Age of the card in days")
    is_foreign_transaction: bool = Field(..., description="Whether transaction is foreign")
    merchant_risk_score: float = Field(..., description="Merchant risk score (0-1)")


class FraudPrediction(BaseModel):
    """Fraud prediction result."""

    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    confidence_score: float = Field(..., description="Model confidence score")
    feature_importance: dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")


class FraudDetector:
    """Fraud detection model using XGBoost."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the fraud detector.

        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path or settings.model_path
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list[str] = []
        self.model_version = settings.model_version
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and scaler."""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(f"Model file not found at {self.model_path}, creating dummy model")
                self._create_dummy_model()
                return

            with open(model_file, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.model_version = model_data.get("version", self.model_version)

            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model version: {self.model_version}")
            logger.info(f"Feature names: {self.feature_names}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating dummy model for development")
            self._create_dummy_model()

    def _create_dummy_model(self) -> None:
        """Create a dummy model for development/testing purposes."""
        # Create dummy data for training
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 10% fraud rate

        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        self.feature_names = [
            "transaction_amount_scaled",
            "merchant_category_encoded",
            "time_of_day_scaled",
            "location_lat_scaled",
            "location_lon_scaled",
            "average_spend_scaled",
            "transactions_last_hour_scaled",
            "card_age_days_scaled",
            "is_foreign_transaction",
            "merchant_risk_score",
        ]

        logger.info("Dummy model created for development")

    def _preprocess_features(self, features: TransactionFeatures) -> np.ndarray:
        """Preprocess transaction features for model input.

        Args:
            features: Transaction features

        Returns
        -------
            Preprocessed feature array
        """
        # Convert to feature vector
        feature_dict = features.model_dump()

        # Create feature array in the expected order
        feature_array = np.array(
            [
                feature_dict["transaction_amount"],
                hash(feature_dict["merchant_category"]) % 1000,  # Simple encoding
                feature_dict["time_of_day"],
                feature_dict["location_lat"],
                feature_dict["location_lon"],
                feature_dict["average_spend"],
                feature_dict["transactions_last_hour"],
                feature_dict["card_age_days"],
                float(feature_dict["is_foreign_transaction"]),
                feature_dict["merchant_risk_score"],
            ]
        ).reshape(1, -1)

        # Scale features
        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)

        return feature_array

    def predict(self, features: TransactionFeatures) -> FraudPrediction:
        """Predict fraud probability for a transaction.

        Args:
            features: Transaction features

        Returns
        -------
            Fraud prediction result
        """
        try:
            # Preprocess features
            X = self._preprocess_features(features)

            # Make prediction
            if self.model is None:
                raise ValueError("Model not loaded")

            fraud_probability = float(self.model.predict_proba(X)[0, 1])
            is_fraud = fraud_probability > settings.prediction_threshold

            # Calculate confidence score (distance from threshold)
            confidence_score = abs(fraud_probability - settings.prediction_threshold)

            # Get feature importance if available
            feature_importance = {}
            if hasattr(self.model, "feature_importances_"):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

            return FraudPrediction(
                fraud_probability=fraud_probability,
                is_fraud=is_fraud,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                model_version=self.model_version,
                prediction_timestamp=pd.Timestamp.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def batch_predict(self, features_list: list[TransactionFeatures]) -> list[FraudPrediction]:
        """Make batch predictions for multiple transactions.

        Args:
            features_list: List of transaction features

        Returns
        -------
            List of fraud predictions
        """
        predictions = []
        for features in features_list:
            try:
                prediction = self.predict(features)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error in batch prediction: {e}")
                # Return a default prediction for failed cases
                predictions.append(
                    FraudPrediction(
                        fraud_probability=0.5,
                        is_fraud=False,
                        confidence_score=0.0,
                        feature_importance={},
                        model_version=self.model_version,
                        prediction_timestamp=pd.Timestamp.now().isoformat(),
                    )
                )

        return predictions

    def get_model_info(self) -> dict[str, Any]:
        """Get model information and metadata.

        Returns
        -------
            Model information dictionary
        """
        return {
            "model_version": self.model_version,
            "model_path": self.model_path,
            "feature_names": self.feature_names,
            "prediction_threshold": settings.prediction_threshold,
            "model_type": type(self.model).__name__ if self.model else "None",
            "is_dummy_model": self.model is not None and isinstance(self.model, RandomForestClassifier),
        }
