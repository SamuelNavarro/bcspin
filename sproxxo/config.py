"""Configuration management for Sproxxo MLOps platform."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = Field(default="Sproxxo Fraud Detection API", description="Application name")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port", env="PORT")
    workers: int = Field(default=1, description="Number of workers")

    # Model settings
    model_path: str = Field(
        default="artifacts/fraud_detection_model_latest.pkl", description="Path to the trained model"
    )
    model_version: str = Field(default="v1.0.0", description="Model version")
    prediction_threshold: float = Field(default=0.8, description="Fraud prediction threshold")

    # MLflow settings
    mlflow_tracking_uri: str = Field(default="file:///tmp/mlruns", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="fraud-detection", description="MLflow experiment name")

    # Security settings
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")

    # Performance settings
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(default=100, description="Maximum concurrent requests")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
