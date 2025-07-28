"""Sproxxo Fraud Detection MLOps Platform."""

__version__ = "0.1.0"
__author__ = "Sproxxo Team"
__email__ = "mlops@sproxxo.com"

from . import api, models, monitoring, training, utils

__all__ = ["api", "models", "monitoring", "training", "utils"]
