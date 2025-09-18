"""
TFT Model Implementation
Temporal Fusion Transformer model wrapper with professional interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from pathlib import Path
import warnings
from ..base_model import BaseModel, ForecastResult, ModelNotFittedError, ModelFittingError

# Handle PyTorch Forecasting imports
try:
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_lightning import Trainer
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None


class TftModel(BaseModel):
    """
    Temporal Fusion Transformer (TFT) model wrapper.
    
    Provides a clean interface to the pre-trained TFT model.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 dataset_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize TFT model.
        
        Args:
            model_path: Path to saved TFT model (.pt file)
            dataset_path: Path to saved TimeSeriesDataSet (.pkl file)
        """
        super().__init__(name="TFT", **kwargs)
        
        if not PYTORCH_FORECASTING_AVAILABLE:
            raise ImportError("pytorch-forecasting is required for TFT model")
        
        # Default paths - use models/tft for checkpoints
        if model_path is None:
            model_path = "data/processed/models/tft.pt"
        if dataset_path is None:
            dataset_path = "data/processed/models/tft_ds.pkl"
            
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        
        # TFT checkpoint directory
        self.tft_checkpoint_dir = Path(__file__).parent / "checkpoints"
        
        self.pytorch_model = None
        self.dataset = None
        self.training_data = None
    
    def _load_pretrained_model(self):
        """Load pre-trained TFT model and dataset."""
        try:
            # Load dataset first
            if self.dataset_path.exists():
                import pickle
                with open(self.dataset_path, 'rb') as f:
                    self.dataset = pickle.load(f)
                self.logger.info(f"Loaded TFT dataset from {self.dataset_path}")
            else:
                raise FileNotFoundError(f"TFT dataset not found at {self.dataset_path}")
            
            # Load model - try different approaches
            if self.model_path.exists():
                # Try loading as state dict first
                try:
                    # Create model from dataset
                    from pytorch_forecasting.metrics import QuantileLoss
                    self.pytorch_model = TemporalFusionTransformer.from_dataset(
                        self.dataset,
                        learning_rate=2e-3,
                        hidden_size=32,
                        attention_head_size=4,
                        dropout=0.1,
                        hidden_continuous_size=16,
                        loss=QuantileLoss(),
                        optimizer="Adam"
                    )
                    
                    # Load state dict
                    state_dict = torch.load(self.model_path, map_location="cpu")
                    self.pytorch_model.load_state_dict(state_dict)
                    self.pytorch_model.eval()
                    self.logger.info(f"Loaded TFT model state dict from {self.model_path}")
                    
                except Exception as state_dict_error:
                    # Try loading as Lightning checkpoint
                    try:
                        self.pytorch_model = TemporalFusionTransformer.load_from_checkpoint(str(self.model_path))
                        self.pytorch_model.eval()
                        self.logger.info(f"Loaded TFT model checkpoint from {self.model_path}")
                    except Exception as checkpoint_error:
                        # Try loading from TFT checkpoints directory
                        checkpoint_files = list(self.tft_checkpoint_dir.glob("*.ckpt"))
                        if checkpoint_files:
                            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                            self.pytorch_model = TemporalFusionTransformer.load_from_checkpoint(str(latest_checkpoint))
                            self.pytorch_model.eval()
                            self.logger.info(f"Loaded TFT model from checkpoint {latest_checkpoint}")
                        else:
                            raise ModelFittingError(f"Failed to load TFT model. State dict error: {state_dict_error}, Checkpoint error: {checkpoint_error}")
            else:
                # Try loading from checkpoints directory
                checkpoint_files = list(self.tft_checkpoint_dir.glob("*.ckpt"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    self.pytorch_model = TemporalFusionTransformer.load_from_checkpoint(str(latest_checkpoint))
                    self.pytorch_model.eval()
                    self.logger.info(f"Loaded TFT model from checkpoint {latest_checkpoint}")
                else:
                    raise FileNotFoundError(f"No TFT model found at {self.model_path} or in {self.tft_checkpoint_dir}")
                
        except Exception as e:
            raise ModelFittingError(f"Failed to load TFT model: {str(e)}")
    
    def fit(self, 
            data: Union[pd.DataFrame, pd.Series],
            target_column: str = 'value',
            **kwargs) -> 'TftModel':
        """
        Load pre-trained TFT model (TFT training is handled separately).
        
        Args:
            data: Training data (used for validation only)
            target_column: Target column name
        """
        try:
            # Validate input data
            series = self.validate_data(data)
            self.training_data = series
            
            # Load pre-trained model
            self._load_pretrained_model()
            
            self._is_fitted = True
            
            # Store model information
            self.config.model_info = {
                'model_path': str(self.model_path),
                'dataset_path': str(self.dataset_path),
                'data_length': len(series),
                'model_type': 'temporal_fusion_transformer',
                'pretrained': True
            }
            
            # Extract model hyperparameters if available
            if hasattr(self.pytorch_model, 'hparams'):
                hparams = dict(self.pytorch_model.hparams)
                # Remove complex objects that can't be serialized
                clean_hparams = {}
                for k, v in hparams.items():
                    if isinstance(v, (int, float, str, bool, list)):
                        clean_hparams[k] = v
                self.config.model_info['hyperparameters'] = clean_hparams
            
            self.logger.info("TFT model loaded successfully")
            return self
            
        except Exception as e:
            raise ModelFittingError(f"Failed to fit TFT model: {str(e)}")
    
    def predict(self, 
                steps: int,
                confidence_level: float = 0.95,
                **kwargs) -> ForecastResult:
        """Generate TFT forecasts."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        
        try:
            # Create validation dataloader
            val_dataloader = self.dataset.to_dataloader(
                train=False, 
                batch_size=1, 
                num_workers=0
            )
            
            # Generate predictions
            with torch.no_grad():
                predictions = self.pytorch_model.predict(
                    val_dataloader, 
                    mode="prediction"
                )
            
            # Extract predictions for requested steps
            if isinstance(predictions, torch.Tensor):
                pred_array = predictions.cpu().numpy()
                
                # Handle different prediction shapes
                if pred_array.ndim > 1:
                    # Take first column if multi-dimensional
                    pred_array = pred_array[:, 0] if pred_array.shape[1] > 0 else pred_array.flatten()
                
                # Take last 'steps' predictions
                if len(pred_array) >= steps:
                    final_predictions = pred_array[-steps:]
                else:
                    # Extend with last value if not enough predictions
                    last_val = pred_array[-1] if len(pred_array) > 0 else 0
                    final_predictions = np.concatenate([
                        pred_array,
                        np.full(steps - len(pred_array), last_val)
                    ])
            else:
                # Fallback if predictions format is unexpected
                final_predictions = np.full(steps, 0)
            
            return ForecastResult(
                predictions=final_predictions,
                model_name=self.name,
                metadata={
                    'method': 'temporal_fusion_transformer',
                    'prediction_shape': str(predictions.shape) if hasattr(predictions, 'shape') else 'unknown',
                    'steps_requested': steps,
                    'steps_generated': len(final_predictions)
                }
            )
            
        except Exception as e:
            self.logger.error(f"TFT prediction failed: {str(e)}")
            
            # Fallback: return zeros or last training value
            fallback_value = 0
            if self.training_data is not None and len(self.training_data) > 0:
                fallback_value = self.training_data.iloc[-1]
            
            predictions = np.full(steps, fallback_value)
            
            return ForecastResult(
                predictions=predictions,
                model_name=self.name,
                metadata={
                    'method': 'tft_fallback',
                    'error': str(e),
                    'fallback_value': fallback_value
                }
            )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = {
            'model_path': str(self.model_path),
            'dataset_path': str(self.dataset_path),
            'pretrained': True
        }
        
        if self._is_fitted and hasattr(self.pytorch_model, 'hparams'):
            # Extract key hyperparameters
            hparams = dict(self.pytorch_model.hparams)
            key_params = {}
            
            for key in ['learning_rate', 'hidden_size', 'attention_head_size', 
                       'dropout', 'hidden_continuous_size']:
                if key in hparams:
                    key_params[key] = hparams[key]
            
            params['hyperparameters'] = key_params
        
        return params
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Get detailed model architecture information."""
        if not self._is_fitted:
            return {'error': 'Model not fitted'}
        
        architecture = {}
        
        try:
            if hasattr(self.pytorch_model, 'hparams'):
                hparams = dict(self.pytorch_model.hparams)
                
                architecture.update({
                    'model_type': 'TemporalFusionTransformer',
                    'hidden_size': hparams.get('hidden_size', 'unknown'),
                    'attention_head_size': hparams.get('attention_head_size', 'unknown'),
                    'dropout': hparams.get('dropout', 'unknown'),
                    'num_attention_heads': hparams.get('num_attention_heads', 'unknown'),
                    'learning_rate': hparams.get('learning_rate', 'unknown')
                })
            
            # Model parameter count
            if hasattr(self.pytorch_model, 'parameters'):
                total_params = sum(p.numel() for p in self.pytorch_model.parameters())
                trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
                
                architecture.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                })
                
        except Exception as e:
            architecture['error'] = str(e)
        
        return architecture
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run TFT model diagnostics."""
        if not self._is_fitted:
            return {'error': 'Model not fitted'}
        
        diagnostics = {}
        
        try:
            # Model file information
            diagnostics.update({
                'model_exists': self.model_path.exists(),
                'dataset_exists': self.dataset_path.exists(),
                'model_size_mb': self.model_path.stat().st_size / (1024*1024) if self.model_path.exists() else 0
            })
            
            # Model architecture
            architecture = self.get_model_architecture()
            diagnostics['architecture'] = architecture
            
            # Dataset information
            if self.dataset is not None:
                diagnostics['dataset_info'] = {
                    'type': str(type(self.dataset)),
                    'has_validation': hasattr(self.dataset, 'to_dataloader')
                }
            
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics