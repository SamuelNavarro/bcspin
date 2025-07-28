"""Main FastAPI application for Sproxxo fraud detection API."""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..config import settings
from ..models import FraudDetector, ModelManager, TransactionFeatures
from ..monitoring import setup_logging
from ..utils.validators import (
    get_overall_risk_score,
    get_risk_indicators,
    validate_transaction_features,
)

setup_logging()
logger = structlog.get_logger()

fraud_detector: FraudDetector
model_manager: ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global fraud_detector, model_manager

    logger.info("Starting Sproxxo Fraud Detection API")

    # Initialize models
    fraud_detector = FraudDetector()
    model_manager = ModelManager()

    logger.info("Models initialized successfully")
    logger.info(f"Active model version: {fraud_detector.model_version}")

    yield

    logger.info("Shutting down Sproxxo Fraud Detection API")


app = FastAPI(
    title=settings.app_name,
    description="Example API for Business Case for Spin",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    # Validation results
    is_valid: bool
    validation_errors: dict[str, list[str]]

    # Risk assessment
    risk_indicators: dict[str, float]
    overall_risk_score: float


class BatchFraudDetectionRequest(BaseModel):
    """Request model for batch fraud detection."""

    transactions: list[FraudDetectionRequest]


class BatchFraudDetectionResponse(BaseModel):
    """Response model for batch fraud detection."""

    predictions: list[FraudDetectionResponse]
    total_transactions: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_version: str
    model_loaded: bool
    uptime_seconds: float


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for collecting metrics and logging."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "Request processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
        client_ip=request.client.host if request.client else None,
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path, method=request.method)

    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global fraud_detector

    return HealthResponse(
        status="healthy",
        model_version=fraud_detector.model_version,
        model_loaded=fraud_detector.model is not None,
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
    )


@app.post("/predict", response_model=FraudDetectionResponse)
async def predict_fraud(request: FraudDetectionRequest):
    """Predict fraud for a single transaction with validation and risk assessment."""
    global fraud_detector

    try:
        # Step 1: Validate transaction features
        validation_errors = validate_transaction_features(request.features)
        is_valid = len(validation_errors) == 0

        # Step 2: Get risk indicators
        risk_indicators = get_risk_indicators(request.features)
        overall_risk_score = get_overall_risk_score(request.features)

        # Step 3: Make ML prediction (even if validation fails, for comparison)
        prediction = fraud_detector.predict(request.features)

        if not is_valid:
            logger.warning(
                "Transaction validation failed",
                transaction_id=request.transaction_id,
                validation_errors=validation_errors,
                risk_score=overall_risk_score,
            )

        # Log high-risk transactions
        if overall_risk_score > 0.7:
            logger.warning(
                "High-risk transaction detected",
                transaction_id=request.transaction_id,
                risk_score=overall_risk_score,
                risk_indicators=risk_indicators,
            )

        return FraudDetectionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=prediction.fraud_probability,
            is_fraud=prediction.is_fraud,
            confidence_score=prediction.confidence_score,
            model_version=prediction.model_version,
            prediction_timestamp=prediction.prediction_timestamp,
            feature_importance=prediction.feature_importance,
            is_valid=is_valid,
            validation_errors=validation_errors,
            risk_indicators=risk_indicators,
            overall_risk_score=overall_risk_score,
        )

    except Exception as e:
        logger.error("Error making prediction", transaction_id=request.transaction_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error making prediction") from None


@app.post("/predict/batch", response_model=BatchFraudDetectionResponse)
async def predict_fraud_batch(request: BatchFraudDetectionRequest):
    """Predict fraud for multiple transactions with validation and risk assessment."""
    global fraud_detector

    try:
        start_time = time.time()

        # Extract features
        features_list = [req.features for req in request.transactions]

        # Make batch predictions
        predictions = fraud_detector.batch_predict(features_list)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        fraud_count = sum(1 for p in predictions if p.is_fraud)

        # Track validation and risk metrics
        invalid_count = 0
        high_risk_count = 0

        # Create response with validation and risk assessment
        response_predictions = []
        for _i, (req, pred) in enumerate(zip(request.transactions, predictions)):
            # Validate each transaction
            validation_errors = validate_transaction_features(req.features)
            is_valid = len(validation_errors) == 0

            # Get risk indicators
            risk_indicators = get_risk_indicators(req.features)
            overall_risk_score = get_overall_risk_score(req.features)

            # Count invalid and high-risk transactions
            if not is_valid:
                invalid_count += 1
            if overall_risk_score > 0.7:
                high_risk_count += 1

            response_predictions.append(
                FraudDetectionResponse(
                    transaction_id=req.transaction_id,
                    fraud_probability=pred.fraud_probability,
                    is_fraud=pred.is_fraud,
                    confidence_score=pred.confidence_score,
                    model_version=pred.model_version,
                    prediction_timestamp=pred.prediction_timestamp,
                    feature_importance=pred.feature_importance,
                    # Validation results
                    is_valid=is_valid,
                    validation_errors=validation_errors,
                    # Risk assessment
                    risk_indicators=risk_indicators,
                    overall_risk_score=overall_risk_score,
                )
            )

        # Log batch processing summary
        logger.info(
            "Batch prediction completed",
            total_transactions=len(request.transactions),
            fraud_count=fraud_count,
            invalid_count=invalid_count,
            high_risk_count=high_risk_count,
            processing_time_ms=processing_time,
        )

        return BatchFraudDetectionResponse(
            predictions=response_predictions,
            total_transactions=len(request.transactions),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error("Error making batch prediction", error=str(e), transaction_count=len(request.transactions))
        raise HTTPException(status_code=500, detail="Error making batch prediction") from None


@app.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    global fraud_detector

    return fraud_detector.get_model_info()


@app.get("/models")
async def list_models():
    """List all available model versions."""
    global model_manager

    models = model_manager.list_models()
    return {"models": [model.model_dump() for model in models], "active_model": fraud_detector.model_version}


@app.get("/models/{version}")
async def get_model_version_info(version: str):
    """Get information about a specific model version."""
    global model_manager

    model_info = model_manager.get_model_info(version)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model version not found")

    return model_info.model_dump()


@app.post("/models/{version}/activate")
async def activate_model(version: str):
    """Activate a specific model version."""
    global model_manager, fraud_detector

    success = model_manager.activate_model(version)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to activate model")

    # Update global fraud detector
    active_model = model_manager.get_active_model()
    if active_model is not None:
        fraud_detector = active_model

    logger.info(f"Model version {version} activated")

    return {"message": f"Model version {version} activated successfully"}


@app.delete("/models/{version}")
async def delete_model(version: str):
    """Delete a model version."""
    global model_manager

    success = model_manager.delete_model(version)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete model")

    return {"message": f"Model version {version} deleted successfully"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "description": "Real-time fraud detection API for Sproxxo financial services",
        "docs": "/docs",
        "health": "/health",
    }


def main():
    """Run the application."""
    import uvicorn

    uvicorn.run(
        "sproxxo.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )


if __name__ == "__main__":
    main()
