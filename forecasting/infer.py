# forecasting/infer.py
"""
Inference module for the chatbot API.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def forecast_months(h: int = 6):
    """
    Main forecasting function for the chatbot API.
    
    Returns:
        tuple: (forecast_df, events_list)
    """
    print(f"Generating {h}-month forecast...")
    
    # For now, create a simple baseline forecast
    forecast_df = create_baseline_forecast(h)
    events = []  # Empty events for now
    
    print("Forecast completed using baseline method")
    return forecast_df, events

def create_baseline_forecast(h: int = 6):
    """Create a simple baseline forecast using recent data"""
    try:
        # Try to load recent data
        data_path = Path("data/processed/models/training_frame.csv")
        if not data_path.exists():
            data_path = Path("data/raw/ons_vacancies_ratio_total.csv")
            
        if data_path.exists():
            if "training_frame" in str(data_path):
                df = pd.read_csv(data_path)
                df["date"] = pd.to_datetime(df["date"])
            else:
                # Handle raw ONS data
                df = pd.read_csv(data_path, skiprows=8, header=None, names=["date", "value"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"]).copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                df = df.dropna(subset=["date"]).sort_values("date")
            
            # Create forecast
            last_value = df["value"].iloc[-1]
            last_date = df["date"].max()
            
        else:
            # If no data, use a reasonable default
            last_value = 2.3  # Recent UK job vacancy ratio
            last_date = pd.Timestamp.now().replace(day=1)
        
        # Generate future dates
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, h+1)]
        
        # Simple forecast with slight variation
        predictions = [last_value + np.random.normal(0, 0.05) for _ in range(h)]
        predictions = [max(0, pred) for pred in predictions]  # Ensure non-negative
        
        forecast_df = pd.DataFrame({
            "date": future_dates,
            "prediction": predictions
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"Error creating baseline forecast: {e}")
        # Emergency fallback
        future_dates = [pd.Timestamp.now().replace(day=1) + pd.DateOffset(months=i) for i in range(1, h+1)]
        return pd.DataFrame({
            "date": future_dates,
            "prediction": [2.3] * h  # Default reasonable value
        })

if __name__ == "__main__":
    # Test the function
    forecast_df, events = forecast_months(3)
    print("\nForecast:")
    print(forecast_df)
    print(f"\nEvents found: {len(events)}")