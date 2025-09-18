"""
Evaluation Package
Professional model evaluation and comparison framework.
"""

from .metrics import MetricsCalculator, ModelMetrics
from .evaluator import ModelEvaluator, EvaluationResult
from .cross_validation import CrossValidator, CVResult

__all__ = [
    'MetricsCalculator',
    'ModelMetrics', 
    'ModelEvaluator',
    'EvaluationResult',
    'CrossValidator',
    'CVResult'
]