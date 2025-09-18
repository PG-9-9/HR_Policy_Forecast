"""
Models Package
Professional time series forecasting models with consistent interfaces.
"""

from .base_model import BaseModel
from .naive_models import NaiveModel, SeasonalNaiveModel, LinearTrendModel
from .arima_model import ArimaModel
from .ets_model import EtsModel
from .tft.tft_model import TftModel

__all__ = [
    'BaseModel',
    'NaiveModel', 
    'SeasonalNaiveModel',
    'LinearTrendModel',
    'ArimaModel',
    'EtsModel', 
    'TftModel'
]

# Model registry for easy access
MODEL_REGISTRY = {
    'naive': NaiveModel,
    'seasonal_naive': SeasonalNaiveModel,
    'linear_trend': LinearTrendModel,
    'arima': ArimaModel,
    'ets': EtsModel,
    'tft': TftModel
}

def get_model(model_name: str, **kwargs):
    """Factory function to get model by name."""
    if model_name.lower() not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return MODEL_REGISTRY[model_name.lower()](**kwargs)