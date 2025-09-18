"""
Model Evaluator Module
Professional model evaluation and comparison framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings
import time

from models import BaseModel, get_model, MODEL_REGISTRY
from .metrics import MetricsCalculator, ModelMetrics


@dataclass
class EvaluationResult:
    """Container for comprehensive evaluation results."""
    model_metrics: List[ModelMetrics]
    comparison_df: pd.DataFrame
    best_model: ModelMetrics
    evaluation_time: float
    test_data: pd.DataFrame
    train_data: pd.DataFrame
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_metrics': [m.to_dict() for m in self.model_metrics],
            'comparison_df': self.comparison_df.to_dict('records'),
            'best_model': self.best_model.to_dict(),
            'evaluation_time': self.evaluation_time,
            'metadata': self.metadata
        }


class ModelEvaluator:
    """
    Professional model evaluator for time series forecasting.
    
    Provides comprehensive model comparison with proper train/test splits,
    metrics calculation, and result reporting.
    """
    
    def __init__(self,
                 data_path: Optional[str] = None,
                 test_size: int = 6,
                 primary_metric: str = 'smape',
                 random_seed: int = 42):
        """
        Initialize model evaluator.
        
        Args:
            data_path: Path to data file (CSV)
            test_size: Number of periods for test set
            primary_metric: Primary metric for model ranking
            random_seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.primary_metric = primary_metric
        self.random_seed = random_seed
        
        # Components
        self.metrics_calculator = MetricsCalculator(primary_metric=primary_metric)
        self.logger = logging.getLogger(__name__)
        
        # Data
        self.data = None
        self.train_data = None
        self.test_data = None
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Load data if path provided
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate time series data."""
        try:
            # Load data
            data_path = Path(data_path)
            
            if not data_path.exists():
                # Try relative to models directory
                alt_path = Path("data/processed/models") / "training_frame.csv"
                if alt_path.exists():
                    data_path = alt_path
                else:
                    raise FileNotFoundError(f"Data file not found: {data_path}")
            
            self.data = pd.read_csv(data_path)
            
            # Validate required columns
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data = self.data.sort_values('date').reset_index(drop=True)
            
            if 'value' not in self.data.columns:
                # Try to find numeric column
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 1:
                    self.data['value'] = self.data[numeric_cols[0]]
                else:
                    raise ValueError("No 'value' column found and cannot auto-detect target column")
            
            self.logger.info(f"Loaded data: {len(self.data)} observations from {data_path}")
            return self.data
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
    
    def prepare_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare train/test split."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if len(self.data) <= self.test_size:
            raise ValueError(f"Data length ({len(self.data)}) must be greater than test_size ({self.test_size})")
        
        # Split data
        split_idx = len(self.data) - self.test_size
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        self.logger.info(f"Train/test split: {len(self.train_data)} / {len(self.test_data)} observations")
        
        return self.train_data, self.test_data
    
    def evaluate_single_model(self,
                             model_name: str,
                             model_params: Optional[Dict[str, Any]] = None) -> ModelMetrics:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of model to evaluate
            model_params: Optional model parameters
            
        Returns:
            ModelMetrics object with evaluation results
        """
        if self.train_data is None or self.test_data is None:
            self.prepare_train_test_split()
        
        model_params = model_params or {}
        
        try:
            # Get model instance
            model = get_model(model_name, **model_params)
            
            self.logger.info(f"Evaluating {model_name} model...")
            
            # Fit model
            start_time = time.time()
            model.fit(self.train_data, target_column='value')
            fit_time = time.time() - start_time
            
            # Generate predictions
            start_time = time.time()
            forecast_result = model.predict(steps=len(self.test_data))
            predict_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                y_true=self.test_data['value'].values,
                y_pred=forecast_result.predictions,
                model_name=model_name,
                y_train=self.train_data['value'].values,
                confidence_intervals=forecast_result.confidence_intervals
            )
            
            # Add timing information to metadata
            if not hasattr(metrics, 'metadata'):
                metrics.metadata = {}
            metrics.metadata.update({
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time
            })
            
            self.logger.info(f"{model_name} evaluation completed: {self.primary_metric}={getattr(metrics, self.primary_metric):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
            
            # Return error metrics
            return ModelMetrics(
                model_name=f"{model_name}_ERROR",
                mae=float('inf'),
                rmse=float('inf'),
                smape=float('inf'),
                mase=float('inf')
            )
    
    def evaluate_all_models(self,
                           model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                           include_models: Optional[List[str]] = None) -> EvaluationResult:
        """
        Evaluate all available models or specified subset.
        
        Args:
            model_configs: Dictionary of model_name -> parameters
            include_models: List of model names to include
            
        Returns:
            EvaluationResult with comprehensive evaluation
        """
        start_time = time.time()
        
        # Prepare data
        if self.train_data is None or self.test_data is None:
            self.prepare_train_test_split()
        
        # Determine which models to evaluate
        if include_models:
            models_to_test = include_models
        else:
            models_to_test = list(MODEL_REGISTRY.keys())
        
        # Default model configurations
        if model_configs is None:
            model_configs = {
                'naive': {},
                'seasonal_naive': {'season_length': 12},
                'linear_trend': {},
                'arima': {'auto_order': True},
                'ets': {'auto_config': True},
                'tft': {}
            }
        
        self.logger.info(f"Starting evaluation of {len(models_to_test)} models...")
        
        # Evaluate each model
        model_metrics = []
        for model_name in models_to_test:
            if model_name in MODEL_REGISTRY:
                params = model_configs.get(model_name, {})
                metrics = self.evaluate_single_model(model_name, params)
                model_metrics.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = self.metrics_calculator.compare_models(model_metrics)
        
        # Find best model
        best_model = self.metrics_calculator.get_best_model(model_metrics)
        
        # Calculate total evaluation time
        total_time = time.time() - start_time
        
        # Create result object
        result = EvaluationResult(
            model_metrics=model_metrics,
            comparison_df=comparison_df,
            best_model=best_model,
            evaluation_time=total_time,
            test_data=self.test_data.copy(),
            train_data=self.train_data.copy(),
            metadata={
                'test_size': self.test_size,
                'primary_metric': self.primary_metric,
                'models_evaluated': len(model_metrics),
                'data_length': len(self.data) if self.data is not None else 0,
                'random_seed': self.random_seed
            }
        )
        
        self.logger.info(f"Evaluation completed in {total_time:.2f}s. Best model: {best_model.model_name}")
        
        return result
    
    def quick_comparison(self, 
                        models: List[str] = None,
                        display: bool = True) -> pd.DataFrame:
        """
        Quick model comparison with minimal configuration.
        
        Args:
            models: List of model names to compare
            display: Whether to print results
            
        Returns:
            Comparison DataFrame
        """
        if models is None:
            models = ['naive', 'linear_trend', 'arima', 'ets']
        
        result = self.evaluate_all_models(include_models=models)
        
        if display:
            print("\n" + "="*60)
            print(" MODEL COMPARISON RESULTS")
            print("="*60)
            print(result.comparison_df.to_string(index=False, float_format='%.4f'))
            print(f"\nBest Model: {result.best_model.model_name}")
            print(f"Evaluation Time: {result.evaluation_time:.2f}s")
            print("="*60)
        
        return result.comparison_df
    
    def get_prediction_comparison(self) -> pd.DataFrame:
        """Get DataFrame comparing actual vs predicted values for all models."""
        if self.train_data is None or self.test_data is None:
            raise ValueError("No evaluation data available")
        
        # This would be implemented to show actual vs predicted plots
        # For now, return test data structure
        comparison_data = self.test_data.copy()
        
        return comparison_data
    
    def save_results(self, result: EvaluationResult, output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        result.comparison_df.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Save detailed metrics
        detailed_metrics = pd.DataFrame([m.to_dict() for m in result.model_metrics])
        detailed_metrics.to_csv(output_path / "detailed_metrics.csv", index=False)
        
        self.logger.info(f"Results saved to {output_path}")


class CrossValidator:
    """Time series cross-validation for robust model evaluation."""
    
    def __init__(self, 
                 n_splits: int = 3,
                 test_size: int = 6,
                 step_size: int = 1):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV splits
            test_size: Size of each test set
            step_size: Step size between splits
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.step_size = step_size
    
    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test splits for cross-validation."""
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split indices
            test_end = len(data) - i * self.step_size
            test_start = test_end - self.test_size
            
            if test_start <= 0:
                break
            
            train_data = data.iloc[:test_start].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            splits.append((train_data, test_data))
        
        return splits


@dataclass
class CVResult:
    """Container for cross-validation results."""
    cv_metrics: List[List[ModelMetrics]]  # metrics for each split
    mean_metrics: List[ModelMetrics]      # averaged metrics
    std_metrics: Dict[str, Dict[str, float]]  # standard deviations
    best_model: str
    n_splits: int