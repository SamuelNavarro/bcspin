"""Model management for Sproxxo fraud detection."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..config import settings
from .fraud_detector import FraudDetector, TransactionFeatures

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""

    version: str
    model_path: str
    created_at: datetime
    performance_metrics: dict[str, float]
    feature_names: list[str]
    hyperparameters: dict[str, Any]
    training_data_info: dict[str, Any]
    model_type: str


class ModelManager:
    """Manager for fraud detection models."""

    def __init__(self, models_dir: str = "artifacts"):
        """Initialize the model manager.

        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.experiment_name = settings.mlflow_experiment_name

        # Current active model
        self.active_model: Optional[FraudDetector] = None

        self.model_metadata: dict[str, ModelMetadata] = {}

        self._load_metadata()

        if not self.model_metadata:
            logger.warning("No models found. Creating dummy model for development.")
            self.active_model = FraudDetector()

    def _create_sample_input_example(self, model: FraudDetector) -> np.ndarray:
        """Create a sample input example for MLflow model logging.

        Args:
            model: The fraud detector model

        Returns
        -------
            Sample input array
        """
        sample_features = TransactionFeatures(
            transaction_amount=150.75,
            merchant_category="retail",
            time_of_day=14,
            location_lat=40.7128,
            location_lon=-74.0060,
            average_spend=85.30,
            transactions_last_hour=2,
            card_age_days=365,
            is_foreign_transaction=False,
            merchant_risk_score=0.25,
        )

        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([sample_features.model_dump()])

        merchant_mapping = {"retail": 0, "online": 1, "gas": 2, "restaurant": 3, "travel": 4, "grocery": 5}
        df["merchant_category_encoded"] = df["merchant_category"].map(merchant_mapping).fillna(0)

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

        X = df[feature_columns].values

        # Perfom transformation here. Always a source of pain :(
        if model.scaler is not None:
            X_scaled = model.scaler.transform(X)
            return X_scaled

        return X

    def _save_sample_input_example(self):
        """Save a sample input example as JSON for API documentation."""
        sample_features = TransactionFeatures(
            transaction_amount=150.75,
            merchant_category="retail",
            time_of_day=14,
            location_lat=40.7128,
            location_lon=-74.0060,
            average_spend=85.30,
            transactions_last_hour=2,
            card_age_days=365,
            is_foreign_transaction=False,
            merchant_risk_score=0.25,
        )

        # Save to examples directory
        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)

        with open(examples_dir / "sample_input.json", "w") as f:
            json.dump(sample_features.model_dump(), f, indent=2)

        logger.info("Sample input example saved to examples/sample_input.json")

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        try:
            metadata_file = self.models_dir / "metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    self.model_metadata = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.model_metadata = {}

        # Load the most recent model as active
        if self.model_metadata:
            latest_version = max(self.model_metadata.keys())
            try:
                self.active_model = FraudDetector(self.model_metadata[latest_version].model_path)
                logger.info(f"Loaded model version {latest_version}")
            except Exception as e:
                logger.error(f"Error loading model {latest_version}: {e}")
                self.active_model = FraudDetector()
        else:
            self.active_model = FraudDetector()

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        try:
            metadata_file = self.models_dir / "metadata.pkl"
            with open(metadata_file, "wb") as f:
                pickle.dump(self.model_metadata, f)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def register_model(
        self,
        model: FraudDetector,
        version: str,
        performance_metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        training_data_info: dict[str, Any],
    ) -> None:
        """Register a new model version.

        Args:
            model: Trained fraud detector model
            version: Model version string
            performance_metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            training_data_info: Information about training data
        """
        try:
            model_path = self.models_dir / f"model_{version}.pkl"

            model_data = {
                "model": model.model,
                "scaler": model.scaler,
                "feature_names": model.feature_names,
                "version": version,
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            # Create metadata
            metadata = ModelMetadata(
                version=version,
                model_path=str(model_path),
                created_at=datetime.now(),
                performance_metrics=performance_metrics,
                feature_names=model.feature_names,
                hyperparameters=hyperparameters,
                training_data_info=training_data_info,
                model_type=type(model.model).__name__,
            )

            # Store metadata
            self.model_metadata[version] = metadata

            # Log to MLflow
            self._log_to_mlflow(model, metadata)

            self._save_sample_input_example()

            self._save_metadata()

            logger.info(f"Model version {version} registered successfully")

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def _log_to_mlflow(self, model: FraudDetector, metadata: ModelMetadata) -> None:
        """Log model to MLflow for experiment tracking."""
        try:
            # Set experiment
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(metadata.hyperparameters)

                # Log metrics
                mlflow.log_metrics(metadata.performance_metrics)

                input_example = self._create_sample_input_example(model)

                mlflow.sklearn.log_model(
                    model.model,
                    f"fraud-detection-{metadata.version}",
                    registered_model_name="fraud-detection",
                    input_example=input_example,
                )

                mlflow.log_dict(metadata.model_dump(), f"metadata_{metadata.version}.json")

                mlflow.log_artifact("examples/sample_input.json", "input_examples")

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise

    def activate_model(self, version: str) -> bool:
        """Activate a specific model version.

        Args:
            version: Model version to activate

        Returns
        -------
            True if activation successful, False otherwise
        """
        if version not in self.model_metadata:
            logger.error(f"Model version {version} not found")
            return False

        try:
            model_path = self.model_metadata[version].model_path
            self.active_model = FraudDetector(model_path)
            logger.info(f"Activated model version {version}")
            return True
        except Exception as e:
            logger.error(f"Error activating model {version}: {e}")
            return False

    def get_active_model(self) -> Optional[FraudDetector]:
        """Get the currently active model.

        Returns
        -------
            Active fraud detector model or None if no model is active
        """
        return self.active_model

    def list_models(self) -> list[ModelMetadata]:
        """List all registered models.

        Returns
        -------
            List of model metadata
        """
        return list(self.model_metadata.values())

    def get_model_info(self, version: str) -> Optional[ModelMetadata]:
        """Get information about a specific model version.

        Args:
            version: Model version

        Returns
        -------
            Model metadata or None if not found
        """
        return self.model_metadata.get(version)

    def delete_model(self, version: str) -> bool:
        """Delete a specific model version.

        Args:
            version: Model version to delete

        Returns
        -------
            True if deletion successful, False otherwise
        """
        if version not in self.model_metadata:
            logger.error(f"Model version {version} not found")
            return False

        try:
            model_path = Path(self.model_metadata[version].model_path)
            if model_path.exists():
                model_path.unlink()

            del self.model_metadata[version]

            self._save_metadata()

            logger.info(f"Deleted model version {version}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {version}: {e}")
            return False

    def export_model(self, version: str, export_path: str) -> bool:
        """Export a model version to a specified path.

        Args:
            version: Model version to export
            export_path: Path to export the model to

        Returns
        -------
            True if export successful, False otherwise
        """
        if version not in self.model_metadata:
            logger.error(f"Model version {version} not found")
            return False

        try:
            import shutil

            model_path = Path(self.model_metadata[version].model_path)
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(model_path, export_path_obj)
            logger.info(f"Exported model version {version} to {export_path_obj}")
            return True
        except Exception as e:
            logger.error(f"Error exporting model {version}: {e}")
            return False
