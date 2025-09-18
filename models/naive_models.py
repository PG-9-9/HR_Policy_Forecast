"""
Naive Forecasting Models
Simple baseline models for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from .base_model import BaseModel, ForecastResult, ModelNotFittedError, ModelFittingError


class NaiveModel(BaseModel):
    """
    Naive forecasting model that predicts the last observed value.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Naive", **kwargs)
        self.last_value = None
    
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'NaiveModel':
        """Fit the naive model (just store the last value)."""
        try:
            series = self.validate_data(data)
            self.last_value = series.iloc[-1]
            self._is_fitted = True
            
            self.config.model_info = {
                'last_value': float(self.last_value),
                'data_length': len(series)
            }
            
            self.logger.info(f"Naive model fitted with last value: {self.last_value:.4f}")
            return self
            
        except Exception as e:
            raise ModelFittingError(f"Failed to fit Naive model: {str(e)}")
    
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """Generate naive forecasts (constant value)."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        predictions = np.full(steps, self.last_value)
        
        return ForecastResult(
            predictions=predictions,
            model_name=self.name,
            metadata={
                'method': 'naive',
                'last_value': self.last_value
            }
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'last_value': self.last_value
        }


class SeasonalNaiveModel(BaseModel):
    """
    Seasonal naive model that predicts using same period from previous season.
    """
    
    def __init__(self, season_length: int = 12, **kwargs):
        super().__init__(name="Seasonal_Naive", **kwargs)
        self.season_length = season_length
        self.seasonal_values = None
        self.data_series = None
    
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'SeasonalNaiveModel':
        """Fit the seasonal naive model."""
        try:
            series = self.validate_data(data)
            self.data_series = series
            
            if len(series) >= self.season_length:
                # Get last season's values
                self.seasonal_values = series.iloc[-self.season_length:].values
            else:
                # Fallback to repeating available data
                self.seasonal_values = series.values
            
            self._is_fitted = True
            
            self.config.model_info = {
                'season_length': self.season_length,
                'seasonal_values': self.seasonal_values.tolist(),
                'data_length': len(series)
            }
            
            self.logger.info(f"Seasonal Naive model fitted with season length: {self.season_length}")
            return self
            
        except Exception as e:
            raise ModelFittingError(f"Failed to fit Seasonal Naive model: {str(e)}")
    
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """Generate seasonal naive forecasts."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        predictions = []
        for i in range(steps):
            # Cycle through seasonal values
            seasonal_idx = i % len(self.seasonal_values)
            predictions.append(self.seasonal_values[seasonal_idx])
        
        predictions = np.array(predictions)
        
        return ForecastResult(
            predictions=predictions,
            model_name=self.name,
            metadata={
                'method': 'seasonal_naive',
                'season_length': self.season_length
            }
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'season_length': self.season_length,
            'seasonal_values': self.seasonal_values.tolist() if self.seasonal_values is not None else None
        }


class LinearTrendModel(BaseModel):
    """
    Linear trend model that extrapolates the linear trend.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="Linear_Trend", **kwargs)
        self.slope = None
        self.intercept = None
        self.last_point = None
        self.data_length = None
    
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'LinearTrendModel':
        """Fit linear trend to the data."""
        try:
            series = self.validate_data(data)
            
            if len(series) < 2:
                raise ValueError("Need at least 2 data points for linear trend")
            
            # Fit linear trend
            x = np.arange(len(series))
            y = series.values
            
            # Calculate slope and intercept
            self.slope = np.polyfit(x, y, 1)[0]
            self.intercept = np.polyfit(x, y, 1)[1]
            
            self.last_point = len(series) - 1
            self.data_length = len(series)
            
            self._is_fitted = True
            
            self.config.model_info = {
                'slope': float(self.slope),
                'intercept': float(self.intercept),
                'last_point': self.last_point,
                'data_length': self.data_length
            }
            
            self.logger.info(f"Linear Trend model fitted with slope: {self.slope:.6f}")
            return self
            
        except Exception as e:
            raise ModelFittingError(f"Failed to fit Linear Trend model: {str(e)}")
    
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """Generate linear trend forecasts."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        # Predict future points using linear extrapolation
        future_x = np.arange(self.last_point + 1, self.last_point + 1 + steps)
        predictions = self.slope * future_x + self.intercept
        
        return ForecastResult(
            predictions=predictions,
            model_name=self.name,
            metadata={
                'method': 'linear_trend',
                'slope': self.slope,
                'intercept': self.intercept
            }
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'last_point': self.last_point
        }