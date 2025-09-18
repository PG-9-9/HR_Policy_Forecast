"""
ETS Model Implementation
Exponential Smoothing State Space Model with automatic configuration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
import warnings
from .base_model import BaseModel, ForecastResult, ModelNotFittedError, ModelFittingError

# Handle statsmodels import
try:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel as StatsETSModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    StatsETSModel = None


class EtsModel(BaseModel):
    """
    ETS (Error, Trend, Seasonal) Exponential Smoothing model.
    
    Supports automatic model selection and handles various ETS configurations.
    """
    
    def __init__(self, 
                 error: Optional[str] = None,
                 trend: Optional[str] = None,
                 seasonal: Optional[str] = None,
                 auto_config: bool = True,
                 seasonal_periods: Optional[int] = None,
                 **kwargs):
        """
        Initialize ETS model.
        
        Args:
            error: Error component ('add', 'mul', or None for auto)
            trend: Trend component ('add', 'mul', or None for auto) 
            seasonal: Seasonal component ('add', 'mul', or None for auto)
            auto_config: Whether to automatically select best configuration
            seasonal_periods: Number of periods in seasonal cycle
        """
        super().__init__(name="ETS", **kwargs)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ETS model. Install with: pip install statsmodels")
        
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.auto_config = auto_config
        self.seasonal_periods = seasonal_periods
        
        self.fitted_model = None
        self.selected_config = None
        self.aic_score = None
        self.bic_score = None
    
    def _auto_select_config(self, series: pd.Series) -> Tuple[str, str, str]:
        """Automatically select best ETS configuration using AIC."""
        best_aic = np.inf
        best_config = ('add', None, None)
        
        # Determine if we should try seasonal models
        try_seasonal = (self.seasonal_periods is not None and 
                       len(series) >= 2 * (self.seasonal_periods or 12))
        
        # Test different configurations
        error_options = ['add', 'mul'] if self.error is None else [self.error]
        trend_options = ['add', 'mul', None] if self.trend is None else [self.trend]
        seasonal_options = (['add', 'mul', None] if try_seasonal else [None]) if self.seasonal is None else [self.seasonal]
        
        for error in error_options:
            for trend in trend_options:
                for seasonal in seasonal_options:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            model = StatsETSModel(
                                series, 
                                error=error, 
                                trend=trend, 
                                seasonal=seasonal,
                                seasonal_periods=self.seasonal_periods
                            )
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_config = (error, trend, seasonal)
                                
                    except Exception:
                        continue
        
        self.logger.info(f"Auto-selected ETS config: {best_config} (AIC: {best_aic:.2f})")
        return best_config
    
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'EtsModel':
        """Fit ETS model to data."""
        try:
            series = self.validate_data(data)
            
            # Auto-detect seasonal periods if not provided
            if self.seasonal_periods is None and len(series) >= 24:
                # Simple heuristic: assume monthly data if length > 24
                self.seasonal_periods = 12
            
            # Auto-select configuration if needed
            if self.auto_config or any(x is None for x in [self.error, self.trend, self.seasonal]):
                error, trend, seasonal = self._auto_select_config(series)
                self.selected_config = (error, trend, seasonal)
            else:
                self.selected_config = (self.error, self.trend, self.seasonal)
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = StatsETSModel(
                    series,
                    error=self.selected_config[0],
                    trend=self.selected_config[1], 
                    seasonal=self.selected_config[2],
                    seasonal_periods=self.seasonal_periods
                )
                self.fitted_model = model.fit()
            
            self.aic_score = self.fitted_model.aic
            self.bic_score = self.fitted_model.bic
            self._is_fitted = True
            
            # Store model information
            self.config.model_info = {
                'config': self.selected_config,
                'seasonal_periods': self.seasonal_periods,
                'aic': float(self.aic_score),
                'bic': float(self.bic_score),
                'log_likelihood': float(self.fitted_model.llf),
                'data_length': len(series),
                'smoothing_params': self._get_smoothing_params()
            }
            
            config_str = f"({self.selected_config[0]},{self.selected_config[1]},{self.selected_config[2]})"
            self.logger.info(f"ETS{config_str} fitted successfully (AIC: {self.aic_score:.2f})")
            return self
            
        except Exception as e:
            raise ModelFittingError(f"Failed to fit ETS model: {str(e)}")
    
    def _get_smoothing_params(self) -> Dict[str, float]:
        """Extract smoothing parameters from fitted model."""
        params = {}
        try:
            if hasattr(self.fitted_model, 'params'):
                model_params = self.fitted_model.params
                if hasattr(model_params, 'smoothing_level'):
                    params['alpha'] = float(model_params.smoothing_level)
                if hasattr(model_params, 'smoothing_trend'):
                    params['beta'] = float(model_params.smoothing_trend)
                if hasattr(model_params, 'smoothing_seasonal'):
                    params['gamma'] = float(model_params.smoothing_seasonal)
        except Exception:
            pass
        return params
    
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """Generate ETS forecasts with confidence intervals."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=steps)
            predictions = forecast_result
            
            # Get confidence intervals if available
            conf_int = None
            try:
                forecast_obj = self.fitted_model.get_forecast(steps=steps)
                conf_int_df = forecast_obj.conf_int(alpha=1-confidence_level)
                
                if len(conf_int_df.columns) >= 2:
                    lower = conf_int_df.iloc[:, 0].values
                    upper = conf_int_df.iloc[:, 1].values
                    conf_int = (lower, upper)
            except Exception:
                pass
            
            return ForecastResult(
                predictions=predictions,
                confidence_intervals=conf_int,
                model_name=self.name,
                metadata={
                    'method': 'ets',
                    'config': self.selected_config,
                    'seasonal_periods': self.seasonal_periods,
                    'aic': self.aic_score,
                    'confidence_level': confidence_level
                }
            )
            
        except Exception as e:
            self.logger.error(f"ETS prediction failed: {str(e)}")
            # Fallback: return last value
            last_value = self.fitted_model.fittedvalues.iloc[-1] if hasattr(self.fitted_model, 'fittedvalues') else 0
            predictions = np.full(steps, last_value)
            
            return ForecastResult(
                predictions=predictions,
                model_name=self.name,
                metadata={
                    'method': 'ets_fallback',
                    'config': self.selected_config,
                    'error': str(e)
                }
            )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'error': self.error,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'auto_config': self.auto_config,
            'seasonal_periods': self.seasonal_periods
        }
        
        if self._is_fitted:
            params.update({
                'selected_config': self.selected_config,
                'aic': self.aic_score,
                'bic': self.bic_score,
                'smoothing_params': self._get_smoothing_params()
            })
        
        return params
    
    def get_components(self) -> Optional[Dict[str, np.ndarray]]:
        """Get ETS components (level, trend, seasonal) if available."""
        if not self._is_fitted:
            return None
            
        components = {}
        try:
            if hasattr(self.fitted_model, 'states'):
                states = self.fitted_model.states
                
                # Level component
                if hasattr(states, 'level'):
                    components['level'] = states.level.values
                
                # Trend component
                if hasattr(states, 'trend'):
                    components['trend'] = states.trend.values
                
                # Seasonal component
                if hasattr(states, 'seasonal'):
                    components['seasonal'] = states.seasonal.values
                    
        except Exception as e:
            self.logger.warning(f"Could not extract components: {str(e)}")
            
        return components if components else None
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        if not self._is_fitted:
            return {'error': 'Model not fitted'}
        
        diagnostics = {}
        
        try:
            # Model fit statistics
            diagnostics.update({
                'aic': float(self.aic_score),
                'bic': float(self.bic_score),
                'log_likelihood': float(self.fitted_model.llf),
                'config': self.selected_config,
                'seasonal_periods': self.seasonal_periods
            })
            
            # Smoothing parameters
            smoothing_params = self._get_smoothing_params()
            if smoothing_params:
                diagnostics['smoothing_params'] = smoothing_params
            
            # Components information
            components = self.get_components()
            if components:
                diagnostics['has_components'] = list(components.keys())
            
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics