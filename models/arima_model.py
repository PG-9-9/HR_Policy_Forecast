"""
ARIMA Model Implementation
Professional ARIMA model with automatic parameter selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
import warnings
from .base_model import BaseModel, ForecastResult, ModelNotFittedError, ModelFittingError

# Handle statsmodels import
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ARIMA = None


class ArimaModel(BaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.
    
    Supports automatic order selection and comprehensive diagnostics.
    """
    
    def __init__(self, 
                 order: Optional[Tuple[int, int, int]] = None,
                 auto_order: bool = True,
                 max_p: int = 3,
                 max_d: int = 2,
                 max_q: int = 3,
                 **kwargs):
        """
        Initialize ARIMA model.
        
        Args:
            order: Manual ARIMA order (p, d, q). If None, auto-select.
            auto_order: Whether to automatically select best order
            max_p: Maximum p value for auto-selection
            max_d: Maximum d value for auto-selection  
            max_q: Maximum q value for auto-selection
        """
        super().__init__(name="ARIMA", **kwargs)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA model. Install with: pip install statsmodels")
        
        self.order = order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        
        self.fitted_model = None
        self.selected_order = None
        self.aic_score = None
        self.bic_score = None
    
    def _check_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """Check if series is stationary using ADF test."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(series.dropna())
                
            is_stationary = result[1] <= 0.05  # p-value threshold
            p_value = result[1]
            
            return is_stationary, p_value
            
        except Exception:
            # If test fails, assume non-stationary
            return False, 1.0
    
    def _auto_select_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """Automatically select best ARIMA order using AIC."""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Test different combinations
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            model = ARIMA(series, order=(p, d, q))
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                
                    except Exception:
                        continue
        
        self.logger.info(f"Auto-selected ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'ArimaModel':
        """Fit ARIMA model to data."""
        try:
            series = self.validate_data(data)
            
            # Check stationarity
            is_stationary, p_value = self._check_stationarity(series)
            
            # Auto-select order if not provided
            if self.auto_order or self.order is None:
                self.selected_order = self._auto_select_order(series)
            else:
                self.selected_order = self.order
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = ARIMA(series, order=self.selected_order)
                self.fitted_model = model.fit()
            
            self.aic_score = self.fitted_model.aic
            self.bic_score = self.fitted_model.bic
            self._is_fitted = True
            
            # Store model information
            self.config.model_info = {
                'order': self.selected_order,
                'aic': float(self.aic_score),
                'bic': float(self.bic_score),
                'is_stationary': is_stationary,
                'stationarity_p_value': float(p_value),
                'log_likelihood': float(self.fitted_model.llf),
                'data_length': len(series)
            }
            
            self.logger.info(f"ARIMA{self.selected_order} fitted successfully (AIC: {self.aic_score:.2f})")
            return self
            
        except Exception as e:
            raise ModelFittingError(f"Failed to fit ARIMA model: {str(e)}")
    
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """Generate ARIMA forecasts with confidence intervals."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model.forecast(
                steps=steps,
                alpha=1-confidence_level  # alpha = 1 - confidence_level
            )
            
            predictions = forecast_result
            
            # Get confidence intervals if available
            conf_int = None
            if hasattr(self.fitted_model, 'get_forecast'):
                forecast_obj = self.fitted_model.get_forecast(steps=steps)
                conf_int_df = forecast_obj.conf_int(alpha=1-confidence_level)
                
                if len(conf_int_df.columns) >= 2:
                    lower = conf_int_df.iloc[:, 0].values
                    upper = conf_int_df.iloc[:, 1].values
                    conf_int = (lower, upper)
            
            return ForecastResult(
                predictions=predictions,
                confidence_intervals=conf_int,
                model_name=self.name,
                metadata={
                    'method': 'arima',
                    'order': self.selected_order,
                    'aic': self.aic_score,
                    'confidence_level': confidence_level
                }
            )
            
        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {str(e)}")
            # Fallback: return last value
            last_value = self.fitted_model.fittedvalues.iloc[-1]
            predictions = np.full(steps, last_value)
            
            return ForecastResult(
                predictions=predictions,
                model_name=self.name,
                metadata={
                    'method': 'arima_fallback',
                    'order': self.selected_order,
                    'error': str(e)
                }
            )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'order': self.selected_order,
            'auto_order': self.auto_order,
            'max_p': self.max_p,
            'max_d': self.max_d,
            'max_q': self.max_q
        }
        
        if self._is_fitted:
            params.update({
                'aic': self.aic_score,
                'bic': self.bic_score,
                'selected_order': self.selected_order
            })
        
        return params
    
    def get_residuals(self) -> Optional[np.ndarray]:
        """Get model residuals for diagnostics."""
        if self._is_fitted and hasattr(self.fitted_model, 'resid'):
            return self.fitted_model.resid.values
        return None
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        if not self._is_fitted:
            return {'error': 'Model not fitted'}
        
        diagnostics = {}
        
        try:
            # Ljung-Box test for residual autocorrelation
            residuals = self.get_residuals()
            if residuals is not None and len(residuals) > 10:
                ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4))
                diagnostics['ljung_box_p_value'] = float(ljung_box['lb_pvalue'].iloc[-1])
                diagnostics['residuals_autocorrelated'] = diagnostics['ljung_box_p_value'] < 0.05
            
            # Model information criteria
            diagnostics.update({
                'aic': float(self.aic_score),
                'bic': float(self.bic_score),
                'log_likelihood': float(self.fitted_model.llf),
                'order': self.selected_order
            })
            
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics