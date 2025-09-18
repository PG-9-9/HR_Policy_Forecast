"""
Cross Validation Module
Time series cross-validation for robust model evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from .metrics import MetricsCalculator, ModelMetrics


@dataclass 
class CVResult:
    """Container for cross-validation results."""
    cv_metrics: List[List[ModelMetrics]]  # metrics for each split
    mean_metrics: List[ModelMetrics]      # averaged metrics
    std_metrics: Dict[str, Dict[str, float]]  # standard deviations
    best_model: str
    n_splits: int


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