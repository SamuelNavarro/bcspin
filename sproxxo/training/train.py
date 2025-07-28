"""Training script for Sproxxo fraud detection model."""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..models import FraudDetector, ModelManager
from ..monitoring import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000, fraud_rate: float = 0.1) -> pd.DataFrame:
    """Generate synthetic fraud detection data for training.

    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent transactions

    Returns
    -------
        DataFrame with synthetic transaction data
    """
    np.random.seed(42)

    # Generate features
    data = {
        "transaction_amount": np.random.exponential(100, n_samples),
        "merchant_category": np.random.choice(
            ["retail", "online", "gas", "restaurant", "travel", "grocery"], n_samples
        ),
        "time_of_day": np.random.randint(0, 24, n_samples),
        "location_lat": np.random.uniform(30, 50, n_samples),
        "location_lon": np.random.uniform(-120, -70, n_samples),
        "average_spend": np.random.exponential(50, n_samples),
        "transactions_last_hour": np.random.poisson(2, n_samples),
        "card_age_days": np.random.exponential(365, n_samples),
        "is_foreign_transaction": np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
        "merchant_risk_score": np.random.beta(2, 5, n_samples),
    }

    df = pd.DataFrame(data)

    # Generate fraud labels based on feature patterns
    fraud_prob = (
        (df["transaction_amount"] > 500) * 0.3
        + (df["is_foreign_transaction"]) * 0.4
        + (df["merchant_risk_score"] > 0.7) * 0.3
        + (df["transactions_last_hour"] > 5) * 0.2
        + (df["time_of_day"].isin([0, 1, 2, 3, 4, 5])) * 0.1
    )

    # Add some randomness
    fraud_prob += np.random.normal(0, 0.1, n_samples)
    fraud_prob = np.clip(fraud_prob, 0, 1)

    # Generate labels
    df["is_fraud"] = np.random.binomial(1, fraud_prob, n_samples)

    # Adjust to target fraud rate
    current_fraud_rate = df["is_fraud"].mean()
    if current_fraud_rate > fraud_rate:
        # Remove some fraud cases
        fraud_indices = df[df["is_fraud"] == 1].index
        if len(fraud_indices) > 0:
            remove_count = int(len(fraud_indices) * (1 - fraud_rate / current_fraud_rate))
            remove_count = max(0, min(remove_count, len(fraud_indices)))
            if remove_count > 0:
                remove_indices = np.random.choice(fraud_indices, remove_count, replace=False)
                df.loc[remove_indices, "is_fraud"] = 0
    elif current_fraud_rate < fraud_rate:
        # Add some fraud cases
        non_fraud_indices = df[df["is_fraud"] == 0].index
        if len(non_fraud_indices) > 0:
            target_fraud_count = int(n_samples * fraud_rate)
            current_fraud_count = df["is_fraud"].sum()
            add_count = target_fraud_count - current_fraud_count
            add_count = max(0, min(add_count, len(non_fraud_indices)))
            if add_count > 0:
                add_indices = np.random.choice(non_fraud_indices, add_count, replace=False)
                df.loc[add_indices, "is_fraud"] = 1

    logger.info(f"Generated {n_samples} samples with {df['is_fraud'].mean():.3f} fraud rate")
    return df


def preprocess_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess features for model training.

    Args:
        df: Input DataFrame

    Returns
    -------
        Tuple of (X, y) arrays
    """
    # Feature engineering
    df_processed = df.copy()

    # Encode merchant category
    df_processed["merchant_category_encoded"] = df_processed["merchant_category"].astype("category").cat.codes

    # Create feature matrix
    feature_columns = [
        "transaction_amount",
        "merchant_category_encoded",
        "time_of_day",
        "location_lat",
        "location_lon",
        "average_spend",
        "transactions_last_hour",
        "card_age_days",
        "is_foreign_transaction",
        "merchant_risk_score",
    ]

    X = df_processed[feature_columns].values
    y = df_processed["is_fraud"].values

    return X, y


def train_xgboost_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[xgb.XGBClassifier, dict[str, float]]:
    """Train simple XGBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns
    -------
        Tuple of (trained_model, metrics)
    """
    # Simple XGBoost model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss")

    logger.info("Training simple XGBoost model...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    logger.info(f"Validation metrics: {metrics}")

    return model, metrics


def train_random_forest_model(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train Random Forest model as fallback.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns
    -------
        Tuple of (trained_model, metrics)
    """
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    logger.info("Training Random Forest model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    logger.info(f"Random Forest validation metrics: {metrics}")

    return model, metrics


def save_model(model: Any, scaler: StandardScaler, feature_names: list, version: str, metrics: dict[str, float]) -> str:
    """Save trained model to disk.

    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        version: Model version
        metrics: Model metrics

    Returns
    -------
        Path to saved model
    """
    # Create artifacts directory (simulating data science team output)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Save model
    model_path = artifacts_dir / f"fraud_detection_model_{version}.pkl"

    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "version": version,
        "metrics": metrics,
        "created_at": datetime.now().isoformat(),
        "training_framework": "scikit-learn",
        "model_type": type(model).__name__,
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {model_path}")

    # Also create a symlink to latest model for easy access
    latest_model_path = artifacts_dir / "fraud_detection_model_latest.pkl"
    if latest_model_path.exists():
        latest_model_path.unlink()

    # Create symlink (or copy on Windows)
    latest_model_path.symlink_to(model_path.name)

    logger.info(f"Latest model symlink created at {latest_model_path}")

    return str(model_path)


def train_model(
    n_samples: int = 10000, fraud_rate: float = 0.1, model_type: str = "xgboost", version: Optional[str] = None
) -> str:
    """Train a fraud detection model.

    Args:
        n_samples: Number of training samples
        fraud_rate: Proportion of fraudulent transactions
        model_type: Type of model to train (xgboost/random_forest)
        version: Model version string

    Returns
    -------
        Path to saved model
    """
    # Generate version if not provided
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Starting model training - Type: {model_type}, Version: {version}")

    # Generate synthetic data
    df = generate_synthetic_data(n_samples, fraud_rate)

    # Preprocess features
    X, y = preprocess_features(df)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_type.lower() == "xgboost":
        model, metrics = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
    elif model_type.lower() == "random_forest":
        model, metrics = train_random_forest_model(X_train_scaled, y_train, X_val_scaled, y_val)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Final evaluation on test set
    y_test_pred = model.predict(X_test_scaled)
    y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_f1_score": f1_score(y_test, y_test_pred),
        "test_roc_auc": roc_auc_score(y_test, y_test_pred_proba),
    }

    metrics.update(test_metrics)

    # Print classification report
    logger.info("Test Set Classification Report:")
    logger.info(classification_report(y_test, y_test_pred))

    # Save model
    feature_names = [
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

    model_path = save_model(model, scaler, feature_names, version, metrics)

    # Register with model manager
    try:
        model_manager = ModelManager()
        fraud_detector = FraudDetector(model_path)

        training_data_info = {
            "n_samples": n_samples,
            "fraud_rate": fraud_rate,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        }

        best_params: Union[float, dict[str, Any]] = metrics.get("best_params", {})
        hyperparams = best_params if isinstance(best_params, dict) else {}
        model_manager.register_model(fraud_detector, version, metrics, hyperparams, training_data_info)

        logger.info(f"Model {version} registered successfully")

    except Exception as e:
        logger.warning(f"Failed to register model: {e}")

    return model_path


def main():
    """Train Sproxxo fraud detection model."""
    parser = argparse.ArgumentParser(description="Train Sproxxo fraud detection model")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--fraud-rate", type=float, default=0.1, help="Fraud rate in training data")
    parser.add_argument("--model-type", choices=["xgboost", "random_forest"], default="xgboost", help="Model type")
    parser.add_argument("--version", type=str, help="Model version")

    args = parser.parse_args()

    try:
        model_path = train_model(
            n_samples=args.n_samples, fraud_rate=args.fraud_rate, model_type=args.model_type, version=args.version
        )
        print(f"Training completed successfully. Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
