"""
Metrics Calculation Module
Professional implementation of forecasting evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    mae: float
    rmse: float
    smape: float
    mase: float
    mape: Optional[float] = None
    coverage: Optional[float] = None
    forecast_bias: Optional[float] = None
    directional_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'mae': self.mae,
            'rmse': self.rmse,
            'smape': self.smape,
            'mase': self.mase,
            'mape': self.mape,
            'coverage': self.coverage,
            'forecast_bias': self.forecast_bias,
            'directional_accuracy': self.directional_accuracy
        }
    
    def get_ranking_score(self, primary_metric: str = 'smape') -> float:
        """Get the primary metric for ranking (lower is better)."""
        return getattr(self, primary_metric, float('inf'))


class MetricsCalculator:
    """
    Professional metrics calculator for time series forecasting.
    
    Implements standard forecasting accuracy metrics with proper handling
    of edge cases and mathematical stability.
    """
    
    def __init__(self, 
                 primary_metric: str = 'smape',
                 handle_zeros: bool = True,
                 epsilon: float = 1e-8):
        """
        Initialize metrics calculator.
        
        Args:
            primary_metric: Primary metric for model ranking
            handle_zeros: Whether to handle zero values specially
            epsilon: Small value to avoid division by zero
        """
        self.primary_metric = primary_metric
        self.handle_zeros = handle_zeros
        self.epsilon = epsilon
        
        # Validate primary metric
        valid_metrics = ['mae', 'rmse', 'smape', 'mase', 'mape']
        if primary_metric not in valid_metrics:
            raise ValueError(f"Primary metric must be one of {valid_metrics}")
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        Uses the standard definition: 2 * |y_true - y_pred| / (|y_true| + |y_pred|)
        """
        numerator = 2.0 * np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        
        # Handle zeros in denominator
        if self.handle_zeros:
            denominator = np.where(denominator < self.epsilon, self.epsilon, denominator)
        
        smape_values = numerator / denominator
        
        # Handle any remaining infinities or NaNs
        smape_values = smape_values[np.isfinite(smape_values)]
        
        if len(smape_values) == 0:
            return 0.0
        
        return float(np.mean(smape_values))
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Returns None if y_true contains zeros.
        """
        # Check for zeros in actual values
        if np.any(np.abs(y_true) < self.epsilon):
            return None
        
        mape_values = np.abs((y_true - y_pred) / y_true)
        return float(np.mean(mape_values))
    
    def calculate_mase(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      y_train: Optional[np.ndarray] = None,
                      seasonal_period: int = 1) -> float:
        """
        Calculate Mean Absolute Scaled Error.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_train: Training data for naive forecast baseline
            seasonal_period: Period for seasonal naive baseline
        """
        mae = self.calculate_mae(y_true, y_pred)
        
        # Calculate naive forecast baseline
        if y_train is not None and len(y_train) > seasonal_period:
            # Use seasonal naive baseline
            if seasonal_period > 1:
                naive_errors = []
                for i in range(seasonal_period, len(y_train)):
                    naive_errors.append(abs(y_train[i] - y_train[i - seasonal_period]))
                naive_mae = np.mean(naive_errors) if naive_errors else 1.0
            else:
                # Simple naive baseline (first differences)
                naive_errors = np.abs(np.diff(y_train))
                naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        else:
            # Use in-sample naive baseline
            if len(y_true) > 1:
                naive_errors = np.abs(np.diff(y_true))
                naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
            else:
                naive_mae = 1.0
        
        # Avoid division by zero
        if naive_mae < self.epsilon:
            naive_mae = self.epsilon
        
        return float(mae / naive_mae)
    
    def calculate_coverage(self, 
                          y_true: np.ndarray,
                          confidence_intervals: tuple,
                          confidence_level: float = 0.95) -> float:
        """
        Calculate prediction interval coverage.
        
        Args:
            y_true: Actual values
            confidence_intervals: Tuple of (lower, upper) bounds
            confidence_level: Expected coverage level
        """
        if confidence_intervals is None:
            return None
        
        lower, upper = confidence_intervals
        
        # Check if actual values fall within intervals
        within_interval = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(within_interval)
        
        return float(coverage)
    
    def calculate_forecast_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate forecast bias (mean of residuals)."""
        bias = np.mean(y_pred - y_true)
        return float(bias)
    
    def calculate_directional_accuracy(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct directional predictions).
        """
        if len(y_true) < 2:
            return None
        
        # Calculate actual and predicted directions (up/down)
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        correct_directions = actual_direction == predicted_direction
        accuracy = np.mean(correct_directions)
        
        return float(accuracy)
    
    def calculate_all_metrics(self,
                             y_true: Union[List, np.ndarray, pd.Series],
                             y_pred: Union[List, np.ndarray, pd.Series],
                             model_name: str,
                             y_train: Optional[Union[List, np.ndarray, pd.Series]] = None,
                             confidence_intervals: Optional[tuple] = None,
                             confidence_level: float = 0.95,
                             seasonal_period: int = 1) -> ModelMetrics:
        """
        Calculate all available metrics for a model.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            y_train: Training data for MASE calculation
            confidence_intervals: Tuple of (lower, upper) prediction intervals
            confidence_level: Confidence level for intervals
            seasonal_period: Seasonal period for MASE calculation
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        if len(y_true) == 0:
            raise ValueError("No data available for metrics calculation")
        
        # Calculate core metrics
        mae = self.calculate_mae(y_true, y_pred)
        rmse = self.calculate_rmse(y_true, y_pred)
        smape = self.calculate_smape(y_true, y_pred)
        
        # Convert y_train if provided
        y_train_array = np.array(y_train) if y_train is not None else None
        mase = self.calculate_mase(y_true, y_pred, y_train_array, seasonal_period)
        
        # Optional metrics
        mape = self.calculate_mape(y_true, y_pred)
        coverage = self.calculate_coverage(y_true, confidence_intervals, confidence_level)
        forecast_bias = self.calculate_forecast_bias(y_true, y_pred)
        directional_accuracy = self.calculate_directional_accuracy(y_true, y_pred)
        
        return ModelMetrics(
            model_name=model_name,
            mae=mae,
            rmse=rmse,
            smape=smape,
            mase=mase,
            mape=mape,
            coverage=coverage,
            forecast_bias=forecast_bias,
            directional_accuracy=directional_accuracy
        )
    
    def compare_models(self, metrics_list: List[ModelMetrics]) -> pd.DataFrame:
        """
        Compare multiple models and create a ranking table.
        
        Args:
            metrics_list: List of ModelMetrics objects
            
        Returns:
            DataFrame with model comparison and rankings
        """
        if not metrics_list:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = [m.to_dict() for m in metrics_list]
        df = pd.DataFrame(data)
        
        # Sort by primary metric (lower is better for all our metrics)
        df = df.sort_values(self.primary_metric, ascending=True).reset_index(drop=True)
        
        # Add ranking
        df.insert(0, 'rank', range(1, len(df) + 1))
        
        # Round numeric columns for better display
        numeric_cols = ['mae', 'rmse', 'smape', 'mase', 'mape', 
                       'coverage', 'forecast_bias', 'directional_accuracy']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        return df
    
    def get_best_model(self, metrics_list: List[ModelMetrics]) -> ModelMetrics:
        """Get the best model based on primary metric."""
        if not metrics_list:
            raise ValueError("No metrics provided")
        
        best_metric = min(metrics_list, key=lambda m: m.get_ranking_score(self.primary_metric))
        return best_metric
    
    def calculate_metric_statistics(self, metrics_list: List[ModelMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics across models for each metric."""
        if not metrics_list:
            return {}
        
        stats = {}
        metric_names = ['mae', 'rmse', 'smape', 'mase']
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in metrics_list if getattr(m, metric_name) is not None]
            
            if values:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return stats