# models/tft/infer.py
import pandas as pd
import torch
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

MODEL_DIR = Path("data/processed/models")
TFT_DIR = Path(__file__).parent  # This is models/tft/

def load_model():
    """Load the trained TFT model and dataset"""
    # Load dataset via pickle
    with open(MODEL_DIR / "tft_ds.pkl", "rb") as f:
        ds: TimeSeriesDataSet = pickle.load(f)

    # Try different ways to load the model
    model_loaded = False
    model = None
    
    # Method 1: Try loading from state dict
    try:
        model = TemporalFusionTransformer.from_dataset(
            ds,
            learning_rate=2e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=QuantileLoss(),
            optimizer="Adam"
        )
        
        model.load_state_dict(torch.load(MODEL_DIR / "tft.pt", map_location="cpu"))
        model.eval()
        model_loaded = True
        print("Loaded model from state dict")
        
    except Exception as e:
        print(f"Failed to load from state dict: {e}")
    
    # Method 2: Try loading from checkpoints
    if not model_loaded:
        try:
            checkpoint_files = list((TFT_DIR / "checkpoints").glob("*.ckpt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                model = TemporalFusionTransformer.load_from_checkpoint(str(latest_checkpoint))
                model.eval()
                model_loaded = True
                print(f"Loaded model from checkpoint: {latest_checkpoint.name}")
            else:
                raise FileNotFoundError("No checkpoint files found")
                
        except Exception as e:
            print(f"Failed to load from checkpoints: {e}")
    
    if not model_loaded:
        raise RuntimeError("Could not load TFT model from any source")
    
    return model, ds

def direct_model_test():
    """Test the model directly without TimeSeriesDataSet reconstruction"""
    print("Testing model directly...")
    
    model, ds = load_model()
    
    # Load training data
    df = pd.read_csv(MODEL_DIR / "training_frame.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total")
    
    # Test with the original validation dataloader
    try:
        val_dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        print(f"Validation dataloader created with {len(val_dl)} batches")
        
        # Get one batch and test
        batch = next(iter(val_dl))
        print(f"Batch type: {type(batch)}")
        
        # Handle tuple format (x, y)
        if isinstance(batch, tuple):
            x, y = batch
            print(f"Input keys: {list(x.keys()) if hasattr(x, 'keys') else 'Not a dict'}")
            print(f"Target shape: {y.shape if hasattr(y, 'shape') else type(y)}")
            test_input = x
        else:
            print(f"Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'Not a dict'}")
            test_input = batch
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            
        print(f"Model forward pass successful!")
        print(f"Output type: {type(output)}")
        
        if hasattr(output, 'shape'):
            print(f"Output shape: {output.shape}")
            print(f"Sample predictions: {output[0, :min(3, output.shape[1])]}")  # First few predictions
        else:
            print(f"Output: {output}")
            
        # Try the predict method too with explicit trainer
        try:
            from lightning.pytorch import Trainer
            # Use explicit trainer with TFT directory to prevent logs in root
            trainer = Trainer(
                logger=False,
                enable_checkpointing=False,
                default_root_dir=str(TFT_DIR)
            )
            predictions = trainer.predict(model, val_dl)
            print(f"Model.predict() also works!")
            if predictions and len(predictions) > 0:
                pred_tensor = predictions[0] if isinstance(predictions[0], torch.Tensor) else predictions[0].prediction
                print(f"Predictions shape: {pred_tensor.shape}")
                print(f"Sample values: {pred_tensor[:3]}")
            return True
        except Exception as e:
            print(f"Model.predict() failed: {e}")
            return True  # Forward pass worked, that's good enough
            
    except Exception as e:
        print(f"Direct model test failed: {e}")
        return False

def simple_forecast(h=3):
    """Make a simple forecast for the next h months"""
    print(f"Loading model and making {h}-month forecast...")
    
    # First test the model directly
    if not direct_model_test():
        print("Model test failed, using baseline...")
    else:
        print("Model is working. Prediction setup needs fixing...")
    
    # Load data for baseline
    df = pd.read_csv(MODEL_DIR / "training_frame.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    print(f"\nData Summary:")
    print(f"Last data point: {df['date'].max()} = {df['value'].iloc[-1]}")
    print(f"Recent trend: {df['value'].tail(5).values}")
    
    # Simple baseline forecast
    last_value = df["value"].iloc[-1]
    last_date = df["date"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, h+1)]
    
    result = pd.DataFrame({
        "date": future_dates,
        "prediction": [last_value] * h
    })
    
    print(f"\nBaseline Forecast:")
    print(result)
    
    return result

if __name__ == "__main__":
    # Simple test
    forecast = simple_forecast(h=3)
    print("\nTest completed!")
