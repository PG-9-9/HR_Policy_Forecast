"""
Base Model Interface
Abstract base class defining the common interface for all forecasting models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    name: str
    params: Dict[str, Any]
    fitted: bool = False
    model_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_info is None:
            self.model_info = {}


@dataclass
class ForecastResult:
    """Container for forecast results."""
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    prediction_dates: Optional[pd.DatetimeIndex] = None
    model_name: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    Provides a consistent interface that all models must implement.
    """
    
    def __init__(self, name: str = None, **kwargs):
        """
        Initialize the model.
        
        Args:
            name: Model name for identification
            **kwargs: Model-specific parameters
        """
        self.name = name or self.__class__.__name__
        self.config = ModelConfig(
            name=self.name,
            params=kwargs
        )
        self.fitted_model = None
        self._is_fitted = False
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            data: Training data (DataFrame or Series)
            target_column: Name of target column if DataFrame
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            confidence_level: Confidence level for intervals
            **kwargs: Additional prediction parameters
            
        Returns:
            ForecastResult containing predictions and metadata
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        self.config.params.update(params)
        return self
    
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'fitted': self._is_fitted,
            'params': self.config.params,
            'model_info': self.config.model_info
        }
    
    def validate_data(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Validate and standardize input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Standardized Series
        """
        if isinstance(data, pd.DataFrame):
            if 'value' in data.columns:
                series = data['value']
            elif len(data.columns) == 1:
                series = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame must have 'value' column or single column")
        else:
            series = data
        
        # Check for missing values
        if series.isnull().any():
            self.logger.warning(f"Data contains {series.isnull().sum()} missing values")
        
        # Check for sufficient data
        if len(series) < 3:
            raise ValueError(f"Insufficient data: {len(series)} observations (minimum 3 required)")
        
        return series
    
    def __repr__(self) -> str:
        """String representation of the model."""
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {fitted_status})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass


class ModelNotFittedError(ModelError):
    """Exception raised when trying to predict with unfitted model."""
    pass


class ModelFittingError(ModelError):
    """Exception raised when model fitting fails."""
    pass