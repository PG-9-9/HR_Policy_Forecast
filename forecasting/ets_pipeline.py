"""
ETS Forecasting Pipeline
Focused on ETS (Exponential Smoothing) model training, evaluation, and forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Any, Optional, Tuple
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our professional models
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models import EtsModel, NaiveModel
from evaluation import ModelEvaluator, MetricsCalculator
from results.visualizations import ChartGenerator

# Paths
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/models")
OUTPUT_DIR = Path("data/processed/forecasting_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ETSForecastingPipeline:
    """
    Professional ETS forecasting pipeline.
    """
    
    def __init__(self, 
                 data_path: str = "data/processed/models/training_frame.csv",
                 test_size: int = 6):
        """
        Initialize ETS forecasting pipeline.
        
        Args:
            data_path: Path to processed data CSV file (defaults to training_frame.csv)
            test_size: Number of periods for test set
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        
        # Initialize components
        self.evaluator = None
        self.chart_generator = ChartGenerator()
        
        # Data
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        
        # Models
        self.ets_model = None
        self.baseline_model = None
        
        logger.info("ETS Forecasting Pipeline initialized")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare data for ETS modeling."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load raw data
        self.raw_data = pd.read_csv(self.data_path)
        
        # Prepare data for forecasting
        if 'date' in self.raw_data.columns:
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
            self.raw_data = self.raw_data.sort_values('date').reset_index(drop=True)
        
        # Create processed dataset
        self.processed_data = self.raw_data.copy()
        
        # Ensure we have a 'value' column
        if 'value' not in self.processed_data.columns:
            # Try common column names
            value_candidates = ['total', 'ratio', 'count', 'amount']
            for col in value_candidates:
                if col in self.processed_data.columns:
                    self.processed_data['value'] = self.processed_data[col]
                    break
            else:
                # Use first numeric column
                numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.processed_data['value'] = self.processed_data[numeric_cols[0]]
                else:
                    raise ValueError("No suitable value column found")
        
        # Save processed data
        processed_path = PROCESSED_DIR / "training_frame.csv"
        self.processed_data.to_csv(processed_path, index=False)
        
        logger.info(f"Data prepared: {len(self.processed_data)} observations")
        logger.info(f"Processed data saved to {processed_path}")
        
        return self.processed_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/test sets."""
        if self.processed_data is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        split_idx = len(self.processed_data) - self.test_size
        self.train_data = self.processed_data.iloc[:split_idx].copy()
        self.test_data = self.processed_data.iloc[split_idx:].copy()
        
        logger.info(f"Data split: {len(self.train_data)} train, {len(self.test_data)} test")
        
        return self.train_data, self.test_data
    
    def train_ets_model(self, **ets_params) -> EtsModel:
        """
        Train ETS model with optimal configuration.
        
        Args:
            **ets_params: Additional ETS parameters
            
        Returns:
            Trained ETS model
        """
        if self.train_data is None:
            self.split_data()
        
        logger.info("Training ETS model...")
        
        # Default ETS configuration (auto-select best configuration)
        default_params = {
            'auto_config': True,
            'seasonal_periods': 12  # Assume monthly data
        }
        default_params.update(ets_params)
        
        # Create and fit ETS model
        self.ets_model = EtsModel(**default_params)
        self.ets_model.fit(self.train_data, target_column='value')
        
        # Also train baseline for comparison
        self.baseline_model = NaiveModel()
        self.baseline_model.fit(self.train_data, target_column='value')
        
        logger.info("ETS model training completed")
        
        return self.ets_model
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained ETS model."""
        if self.ets_model is None:
            raise ValueError("Model not trained. Call train_ets_model() first.")
        
        logger.info("Evaluating ETS model...")
        
        # Initialize evaluator with our data
        self.evaluator = ModelEvaluator(
            data_path=str(PROCESSED_DIR / "training_frame.csv"),
            test_size=self.test_size
        )
        
        # Compare ETS vs baseline
        comparison_df = self.evaluator.quick_comparison(
            models=['ets', 'naive'],
            display=True
        )
        
        return {
            'comparison_df': comparison_df,
            'best_model': comparison_df.iloc[0]['model_name'],
            'ets_smape': comparison_df[comparison_df['model_name'] == 'ets']['smape'].iloc[0]
        }
    
    def generate_forecasts(self, horizon: int = 6) -> Dict[str, np.ndarray]:
        """
        Generate forecasts using the trained ETS model.
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Dictionary with forecasts
        """
        if self.ets_model is None:
            raise ValueError("Model not trained. Call train_ets_model() first.")
        
        logger.info(f"Generating {horizon}-period forecasts...")
        
        # Generate ETS forecasts
        ets_forecast = self.ets_model.predict(steps=horizon)
        
        # Generate baseline forecasts for comparison
        baseline_forecast = self.baseline_model.predict(steps=horizon)
        
        forecasts = {
            'ets': ets_forecast.predictions,
            'naive': baseline_forecast.predictions
        }
        
        logger.info("Forecasts generated successfully")
        
        return forecasts
    
    def create_forecast_report(self, 
                             forecasts: Dict[str, np.ndarray],
                             save_charts: bool = True) -> str:
        """
        Create comprehensive forecast report.
        
        Args:
            forecasts: Dictionary of model forecasts
            save_charts: Whether to save charts
            
        Returns:
            Path to saved report
        """
        logger.info("Creating forecast report...")
        
        # Create visualizations
        if save_charts:
            # Forecast comparison chart
            fig = self.chart_generator.create_forecast_comparison_chart(
                actual_data=self.processed_data['value'],
                forecasts=forecasts,
                dates=self.processed_data.get('date'),
                save_path=OUTPUT_DIR / "ets_forecast_comparison.png"
            )
            plt.close(fig)
        
        # Create text report
        report_path = OUTPUT_DIR / "ets_forecast_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("ETS FORECASTING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n")
            f.write(f"Training Period: {len(self.train_data)} observations\n")
            f.write(f"Test Period: {len(self.test_data)} observations\n\n")
            
            f.write("ETS MODEL CONFIGURATION\n")
            f.write("-" * 25 + "\n")
            if self.ets_model:
                params = self.ets_model.get_params()
                for key, value in params.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("FORECAST RESULTS\n")
            f.write("-" * 16 + "\n")
            for model_name, preds in forecasts.items():
                f.write(f"{model_name.upper()} Forecasts:\n")
                for i, pred in enumerate(preds, 1):
                    f.write(f"  Period +{i}: {pred:.4f}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("- Use ETS model for production forecasting\n")
            f.write("- Monitor forecast accuracy over time\n")
            f.write("- Retrain model monthly with new data\n")
            f.write("- Consider seasonal adjustments if patterns change\n")
        
        logger.info(f"Forecast report saved to {report_path}")
        
        return str(report_path)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ETS forecasting pipeline.
        
        Returns:
            Dictionary with all results
        """
        logger.info("Starting complete ETS forecasting pipeline...")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Split data
        self.split_data()
        
        # Step 3: Train ETS model
        self.train_ets_model()
        
        # Step 4: Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Step 5: Generate forecasts
        forecasts = self.generate_forecasts(horizon=6)
        
        # Step 6: Create report
        report_path = self.create_forecast_report(forecasts)
        
        results = {
            'model': self.ets_model,
            'evaluation': evaluation_results,
            'forecasts': forecasts,
            'report_path': report_path,
            'data_shape': self.processed_data.shape,
            'train_test_split': (len(self.train_data), len(self.test_data))
        }
        
        logger.info("Complete ETS forecasting pipeline finished successfully!")
        
        return results


def main():
    """Main function to run ETS forecasting pipeline."""
    print("ETS Forecasting Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ETSForecastingPipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    # Display summary
    print(f"\nPipeline Results:")
    print(f"- Data shape: {results['data_shape']}")
    print(f"- Train/Test split: {results['train_test_split']}")
    print(f"- Best model: {results['evaluation']['best_model']}")
    print(f"- ETS sMAPE: {results['evaluation']['ets_smape']:.4f}")
    print(f"- Report saved: {results['report_path']}")
    
    print("\nETS forecasts for next 6 periods:")
    for i, pred in enumerate(results['forecasts']['ets'], 1):
        print(f"  Period +{i}: {pred:.4f}")


if __name__ == "__main__":
    main()