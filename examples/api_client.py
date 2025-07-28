#!/usr/bin/env python3
"""Example API client for Sproxxo Fraud Detection API."""

import asyncio
import time

import httpx
from pydantic import BaseModel

from sproxxo.models import TransactionFeatures
from sproxxo.utils.data_generator import (
    generate_fraudulent_transaction,
    generate_legitimate_transaction,
    generate_sample_transaction,
)


class FraudDetectionRequest(BaseModel):
    """Request model for fraud detection."""

    transaction_id: str
    features: TransactionFeatures


class FraudDetectionResponse(BaseModel):
    """Response model for fraud detection."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    confidence_score: float
    model_version: str
    prediction_timestamp: str
    feature_importance: dict


class BatchFraudDetectionRequest(BaseModel):
    """Request model for batch fraud detection."""

    transactions: list[FraudDetectionRequest]


class BatchFraudDetectionResponse(BaseModel):
    """Response model for batch fraud detection."""

    predictions: list[FraudDetectionResponse]
    total_transactions: int
    processing_time_ms: float


class SproxxoAPIClient:
    """Client for Sproxxo Fraud Detection API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def health_check(self) -> dict:
        """Check API health.

        Returns
        -------
            Health check response
        """
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def get_model_info(self) -> dict:
        """Get model information.

        Returns
        -------
            Model information
        """
        response = await self.client.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    async def predict_fraud(self, transaction_id: str, features: TransactionFeatures) -> FraudDetectionResponse:
        """Predict fraud for a single transaction.

        Args:
            transaction_id: Transaction identifier
            features: Transaction features

        Returns
        -------
            Fraud prediction response
        """
        request_data = FraudDetectionRequest(transaction_id=transaction_id, features=features)

        response = await self.client.post(f"{self.base_url}/predict", json=request_data.model_dump())
        response.raise_for_status()

        return FraudDetectionResponse(**response.json())

    async def predict_fraud_batch(self, transactions: list[tuple]) -> BatchFraudDetectionResponse:
        """Predict fraud for multiple transactions.

        Args:
            transactions: List of (transaction_id, features) tuples

        Returns
        -------
            Batch fraud prediction response
        """
        request_data = BatchFraudDetectionRequest(
            transactions=[
                FraudDetectionRequest(transaction_id=transaction_id, features=features)
                for transaction_id, features in transactions
            ]
        )

        response = await self.client.post(f"{self.base_url}/predict/batch", json=request_data.model_dump())
        response.raise_for_status()

        return BatchFraudDetectionResponse(**response.json())

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def run_api_client_demo():
    """Demonstrates API usage."""
    client = SproxxoAPIClient()

    try:
        # Health check
        print("🔍 Checking API health...")
        health = await client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Model Version: {health['model_version']}")
        print(f"🔧 Model Loaded: {health['model_loaded']}")
        print()

        # Get model info
        print("📋 Getting model information...")
        model_info = await client.get_model_info()
        print(f"Model Type: {model_info['model_type']}")
        print(f"Model Version: {model_info['model_version']}")
        print(f"Prediction Threshold: {model_info['prediction_threshold']}")
        print()

        # Single prediction examples
        print("🎯 Single Transaction Predictions")
        print("=" * 50)

        # Legitimate transaction
        legitimate_features = generate_legitimate_transaction()
        legitimate_prediction = await client.predict_fraud("txn_001", legitimate_features)
        print("Legitimate Transaction:")
        print(f"  Transaction ID: {legitimate_prediction.transaction_id}")
        print(f"  Fraud Probability: {legitimate_prediction.fraud_probability:.3f}")
        print(f"  Is Fraud: {legitimate_prediction.is_fraud}")
        print(f"  Confidence: {legitimate_prediction.confidence_score:.3f}")
        print()

        # Fraudulent transaction
        fraudulent_features = generate_fraudulent_transaction()
        fraudulent_prediction = await client.predict_fraud("txn_002", fraudulent_features)
        print("Fraudulent Transaction:")
        print(f"  Transaction ID: {fraudulent_prediction.transaction_id}")
        print(f"  Fraud Probability: {fraudulent_prediction.fraud_probability:.3f}")
        print(f"  Is Fraud: {fraudulent_prediction.is_fraud}")
        print(f"  Confidence: {fraudulent_prediction.confidence_score:.3f}")
        print()

        # Batch prediction
        print("📦 Batch Transaction Predictions")
        print("=" * 50)

        # Generate sample transactions
        sample_transactions = []
        for i in range(5):
            features = generate_sample_transaction()
            sample_transactions.append((f"batch_txn_{i + 1:03d}", features))

        batch_prediction = await client.predict_fraud_batch(sample_transactions)
        print(f"Total Transactions: {batch_prediction.total_transactions}")
        print(f"Processing Time: {batch_prediction.processing_time_ms:.2f}ms")
        print()

        print("Predictions:")
        for prediction in batch_prediction.predictions:
            status = "🚨 FRAUD" if prediction.is_fraud else "✅ LEGITIMATE"
            print(f"  {prediction.transaction_id}: {prediction.fraud_probability:.3f} ({status})")
        print()

        # Performance test
        print("⚡ Performance Test")
        print("=" * 50)

        start_time = time.time()
        num_requests = 10

        for i in range(num_requests):
            features = generate_sample_transaction()
            await client.predict_fraud(f"perf_txn_{i + 1}", features)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_requests * 1000  # Convert to milliseconds

        print(f"Requests: {num_requests}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Average Time: {avg_time:.2f}ms")
        print(f"Throughput: {num_requests / total_time:.1f} req/s")
        print()

    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()


def main():
    """Entry point for the API client script."""
    asyncio.run(run_api_client_demo())


if __name__ == "__main__":
    main()
