"""
ETS-Focused Model Comparison Script
Compares ETS model against baseline models for forecasting performance.
"""
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our professional models
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models import EtsModel, NaiveModel, LinearTrendModel, ArimaModel
from evaluation import ModelEvaluator, MetricsCalculator

# Paths
MODEL_DIR = Path("data/processed/models")
OUT_DIR = Path("data/processed/model_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ETSModelComparator:
    """
    Professional ETS-focused model comparison.
    Compares ETS against baseline and traditional statistical models.
    """
    
    def __init__(self, data_path: str = None):
        """Initialize with data"""
        if data_path:
            self.df = pd.read_csv(data_path)
        else:
            # Load training data
            self.df = pd.read_csv(MODEL_DIR / "training_frame.csv")
        
        # Ensure date column exists and is properly formatted
        if 'date' in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"])
        else:
            # Create a date range if missing
            self.df['date'] = pd.date_range(start='2020-01-01', periods=len(self.df), freq='M')
        
        self.df = self.df.sort_values("date").reset_index(drop=True)
        self.results = {}
        
        print(f"[*] Loaded data: {len(self.df)} points from {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Last value: {self.df['value'].iloc[-1]:.2f}")
    
    def compare_all_models(self, test_size: int = 6) -> pd.DataFrame:
        """Compare all available models with focus on ETS"""
        print(f"\n[*] Starting ETS-Focused Model Comparison (test_size={test_size})")
        print("=" * 70)
        
        # Use the professional evaluator
        evaluator = ModelEvaluator(
            data_path=str(MODEL_DIR / "training_frame.csv"),
            test_size=test_size
        )
        
        # Compare ETS against baselines
        models_to_compare = ['ets', 'naive', 'linear_trend', 'arima']
        
        try:
            comparison_df = evaluator.quick_comparison(
                models=models_to_compare,
                display=True
            )
            
            # Store results for additional analysis
            self.results = {
                'comparison_df': comparison_df,
                'evaluator': evaluator
            }
            
            return comparison_df
            
        except Exception as e:
            print(f"[X] Comparison failed: {e}")
            return pd.DataFrame()
    
    def generate_ets_recommendation(self, results_df: pd.DataFrame) -> str:
        """Generate ETS-focused recommendation report"""
        if len(results_df) == 0:
            return "[X] No models were successfully fitted."
        
        best_model = results_df.iloc[0]
        ets_performance = results_df[results_df['model_name'] == 'ets']
        
        recommendation = f"""
*** ETS MODEL ANALYSIS & RECOMMENDATION ***
{'='*60}

[BEST OVERALL MODEL]
{best_model['model_name']} achieved the best performance
  • sMAPE: {best_model['smape']:.4f}
  • RMSE: {best_model['rmse']:.4f}
  • MAE: {best_model['mae']:.4f}
"""
        
        if len(ets_performance) > 0:
            ets_row = ets_performance.iloc[0]
            ets_rank = ets_performance.index[0] + 1
            
            recommendation += f"""
[ETS MODEL PERFORMANCE]
ETS ranked #{ets_rank} out of {len(results_df)} models
  • sMAPE: {ets_row['smape']:.4f}
  • Performance vs best model: {((ets_row['smape']/best_model['smape'])-1)*100:+.1f}%
"""
            
            # ETS-specific insights
            if ets_rank == 1:
                recommendation += """
[TOP PERFORMER] ETS is the best model
  - ETS successfully captured the time series patterns
  - Exponential smoothing is well-suited for this data
  - Use ETS as primary forecasting model
"""
            elif ets_rank <= 2:
                recommendation += """
[GOOD] ETS is among the top performers
  - ETS provides competitive forecasting accuracy
  - Consider ETS for production use
  - The model captures underlying patterns well
"""
            else:
                recommendation += """
[MODERATE] ETS performance could be improved
  - Consider parameter tuning or data preprocessing
  - Simple baselines may be more appropriate
"""
        else:
            recommendation += """
[X] ETS MODEL FAILED
  - ETS could not be fitted to the data
  - Consider data preprocessing or simpler approaches
"""
        
        # Full ranking
        recommendation += "\n\n[COMPLETE MODEL RANKING]\n"
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            medal = "Winner" if i == 1 else "Runner-up" if i == 2 else "Third" if i == 3 else "  "
            ets_indicator = " <- ETS" if row['model_name'] == 'ets' else ""
            recommendation += f"   {medal} #{i} {row['model_name']} (sMAPE: {row['smape']:.4f}){ets_indicator}\n"
        
        # Practical recommendations
        if best_model['model_name'] == 'ets':
            recommendation += """
[PRODUCTION RECOMMENDATIONS]
* Deploy ETS model for forecasting
* Set up automated retraining pipeline
* Monitor forecast accuracy over time
* Consider seasonal adjustments if patterns change
"""
        elif best_model['model_name'] in ['naive', 'linear_trend']:
            recommendation += """
[PRODUCTION RECOMMENDATIONS]
* Consider using simple baseline models
* Time series may be too simple for ETS
* Monitor for pattern changes that might favor ETS
* Simple models are easier to maintain and explain
"""
        else:
            recommendation += """
[PRODUCTION RECOMMENDATIONS]
* Use the best performing model for production
* Keep ETS as a secondary option
* Monitor performance and switch if patterns change
"""
        
        return recommendation
    
    def save_results(self, results_df: pd.DataFrame):
        """Save comparison results"""
        if len(results_df) > 0:
            # Save main results
            results_path = OUT_DIR / "ets_comparison_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"[+] Results saved to {results_path}")
            
            # Generate and save recommendation
            recommendation = self.generate_ets_recommendation(results_df)
            rec_path = OUT_DIR / "ets_recommendation.txt"
            with open(rec_path, "w", encoding='utf-8') as f:
                f.write(recommendation)
            print(f"[+] Recommendation saved to {rec_path}")
            
            return recommendation
        else:
            print("[!] No results to save")
            return ""


def main():
    """Main ETS comparison function"""
    print("[*] ETS-Focused Model Comparison for HR Policy Forecasting")
    print("=" * 70)
    
    # Initialize comparator
    comparator = ETSModelComparator()
    
    # Run comparison
    results_df = comparator.compare_all_models(test_size=6)
    
    if len(results_df) > 0:
        print("\n[*] COMPARISON RESULTS:")
        print("=" * 70)
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save results and get recommendation
        recommendation = comparator.save_results(results_df)
        print("\n" + recommendation)
        
    else:
        print("\nNo models could be successfully fitted and evaluated.")
        print("Please check data and model configurations.")


if __name__ == "__main__":
    main()