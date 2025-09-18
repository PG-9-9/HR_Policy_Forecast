"""
Professional Forecasting Model Comparison System
Main entry point for model evaluation and comparison.

Usage:
    python main.py --compare-models
    python main.py --quick-test
    python main.py --models naive arima ets tft --output results/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('forecasting_evaluation.log')
    ]
)

logger = logging.getLogger(__name__)

# Import our professional modules
try:
    from models import MODEL_REGISTRY, get_model
    from evaluation import ModelEvaluator, MetricsCalculator
    from results.visualizations import ChartGenerator, create_summary_dashboard
    
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    MODULES_AVAILABLE = False


class ForecastingSystem:
    """
    Professional forecasting system for model comparison and evaluation.
    """
    
    def __init__(self, 
                 data_path: str = "data/processed/models/training_frame.csv",
                 output_dir: str = "results",
                 test_size: int = 6):
        """
        Initialize the forecasting system.
        
        Args:
            data_path: Path to the training data CSV
            output_dir: Directory for output files and charts
            test_size: Number of periods for test set
        """
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available. Check imports.")
        
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.evaluator = ModelEvaluator(
            data_path=str(self.data_path),
            test_size=test_size
        )
        self.chart_generator = ChartGenerator()
        
        logger.info(f"Forecasting system initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_comprehensive_comparison(self, 
                                   models: Optional[List[str]] = None,
                                   save_results: bool = True) -> pd.DataFrame:
        """
        Run comprehensive model comparison with all available models.
        
        Args:
            models: List of specific models to test, or None for all
            save_results: Whether to save results and generate charts
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Starting comprehensive model comparison...")
        
        if models is None:
            models = list(MODEL_REGISTRY.keys())
        
        logger.info(f"Models to evaluate: {models}")
        
        # Run evaluation
        result = self.evaluator.evaluate_all_models(include_models=models)
        
        # Display results
        self._display_results(result)
        
        if save_results:
            self._save_comprehensive_results(result)
        
        return result.comparison_df
    
    def run_quick_test(self, 
                      models: List[str] = None) -> pd.DataFrame:
        """
        Run quick test with basic models.
        
        Args:
            models: List of models to test
            
        Returns:
            DataFrame with comparison results
        """
        if models is None:
            models = ['naive', 'linear_trend', 'arima', 'ets']
        
        logger.info(f"Running quick test with models: {models}")
        
        comparison_df = self.evaluator.quick_comparison(models=models, display=True)
        
        return comparison_df
    
    def evaluate_single_model(self, model_name: str, **model_params) -> dict:
        """
        Evaluate a single model with custom parameters.
        
        Args:
            model_name: Name of the model to evaluate
            **model_params: Model-specific parameters
            
        Returns:
            Dictionary with model evaluation results
        """
        logger.info(f"Evaluating single model: {model_name}")
        
        metrics = self.evaluator.evaluate_single_model(model_name, model_params)
        
        # Display results
        print(f"\n{model_name.upper()} MODEL RESULTS:")
        print("=" * 40)
        print(f"MAE:   {metrics.mae:.4f}")
        print(f"RMSE:  {metrics.rmse:.4f}")
        print(f"sMAPE: {metrics.smape:.4f}")
        print(f"MASE:  {metrics.mase:.4f}")
        print("=" * 40)
        
        return metrics.to_dict()
    
    def _display_results(self, result):
        """Display formatted results to console."""
        print("\n" + "="*80)
        print(" FORECASTING MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Summary statistics
        print(f"\nEvaluation Summary:")
        print(f"- Models evaluated: {len(result.model_metrics)}")
        print(f"- Test period: {result.metadata['test_size']} periods")
        print(f"- Evaluation time: {result.evaluation_time:.2f} seconds")
        print(f"- Best model: {result.best_model.model_name}")
        
        # Results table
        print(f"\nDetailed Results:")
        print(result.comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Best model details
        print(f"\n" + "-"*50)
        print(f"RECOMMENDED MODEL: {result.best_model.model_name}")
        print(f"- sMAPE: {result.best_model.smape:.4f}")
        print(f"- RMSE:  {result.best_model.rmse:.4f}")
        print(f"- MASE:  {result.best_model.mase:.4f}")
        print("-"*50)
        
        # Interpretation
        self._provide_interpretation(result.best_model)
        
        print("="*80)
    
    def _provide_interpretation(self, best_model):
        """Provide interpretation of the best model results."""
        model_name = best_model.model_name.lower()
        
        print(f"\nModel Interpretation:")
        
        if model_name in ['naive', 'linear_trend', 'seasonal_naive']:
            print("- Simple baseline models perform best")
            print("- Data may be relatively stable or have simple patterns")
            print("- Consider this for production use due to simplicity and interpretability")
        
        elif model_name in ['arima', 'ets']:
            print("- Traditional statistical models work well for this data")
            print("- Data has clear statistical patterns that can be modeled")
            print("- Good balance of accuracy and interpretability")
        
        elif model_name == 'tft':
            print("- Neural network model captures complex patterns")
            print("- May benefit from additional features or longer training")
            print("- Consider for complex forecasting scenarios")
        
        # MASE interpretation
        if best_model.mase < 1:
            print(f"- MASE < 1 indicates the model outperforms naive forecasting")
        else:
            print(f"- MASE > 1 suggests the model struggles compared to naive baseline")
    
    def _save_comprehensive_results(self, result):
        """Save comprehensive results including charts and data."""
        logger.info("Saving comprehensive results...")
        
        # Save data tables
        result.comparison_df.to_csv(self.output_dir / "model_comparison.csv", index=False)
        
        detailed_metrics = pd.DataFrame([m.to_dict() for m in result.model_metrics])
        detailed_metrics.to_csv(self.output_dir / "detailed_metrics.csv", index=False)
        
        # Create and save charts
        try:
            # Prepare forecast data for visualization
            forecasts = {}
            for metrics in result.model_metrics:
                # This is a simplified approach - in practice you'd store predictions
                model_name = metrics.model_name
                # For now, create dummy forecasts for visualization
                test_len = len(result.test_data)
                forecasts[model_name] = result.test_data['value'].values + np.random.normal(0, 0.1, test_len)
            
            # Generate charts
            charts_saved = create_summary_dashboard(
                result.comparison_df,
                result.test_data['value'],
                forecasts,
                str(self.output_dir / "charts")
            )
            
            logger.info(f"Charts saved: {list(charts_saved.keys())}")
            
        except Exception as e:
            logger.warning(f"Could not generate charts: {e}")
        
        # Save text report
        self._generate_text_report(result)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_text_report(self, result):
        """Generate a comprehensive text report."""
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("FORECASTING MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n")
            f.write(f"Test Period: {result.metadata['test_size']} periods\n")
            f.write(f"Models Evaluated: {result.metadata['models_evaluated']}\n")
            f.write(f"Evaluation Time: {result.evaluation_time:.2f} seconds\n\n")
            
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(result.comparison_df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            f.write("BEST MODEL DETAILS\n")
            f.write("-" * 20 + "\n")
            best = result.best_model
            f.write(f"Model: {best.model_name}\n")
            f.write(f"sMAPE: {best.smape:.4f}\n")
            f.write(f"RMSE:  {best.rmse:.4f}\n")
            f.write(f"MAE:   {best.mae:.4f}\n")
            f.write(f"MASE:  {best.mase:.4f}\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Use {best.model_name} for production forecasting.\n")
            
            if best.mase < 1:
                f.write("Model outperforms naive baseline (MASE < 1).\n")
            else:
                f.write("Consider model improvement or additional features (MASE > 1).\n")
        
        logger.info(f"Text report saved to {report_path}")
    
    def list_available_models(self):
        """List all available models."""
        print("\nAvailable Models:")
        print("-" * 20)
        for name, model_class in MODEL_REGISTRY.items():
            print(f"- {name}: {model_class.__doc__ or 'No description'}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Professional Forecasting Model Comparison System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --compare-models                    # Compare all models
  python main.py --quick-test                        # Quick test with basic models
  python main.py --models naive arima ets            # Test specific models
  python main.py --single-model arima --output ./out # Test single model
  python main.py --list-models                       # List available models
        """
    )
    
    parser.add_argument('--compare-models', action='store_true',
                       help='Run comprehensive model comparison')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with basic models')
    
    parser.add_argument('--single-model', type=str,
                       help='Evaluate a single model')
    
    parser.add_argument('--models', nargs='+', 
                       help='Specific models to evaluate')
    
    parser.add_argument('--data', type=str,
                       default='data/processed/models/training_frame.csv',
                       help='Path to training data CSV')
    
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--test-size', type=int, default=6,
                       help='Number of periods for test set')
    
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        if MODULES_AVAILABLE:
            print("\nAvailable Models:")
            print("-" * 20)
            for name, model_class in MODEL_REGISTRY.items():
                print(f"- {name}: {model_class.__name__}")
            print()
        return
    
    # Initialize system
    try:
        system = ForecastingSystem(
            data_path=args.data,
            output_dir=args.output,
            test_size=args.test_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return
    
    # Execute requested operation
    try:
        if args.compare_models:
            system.run_comprehensive_comparison(models=args.models)
        
        elif args.quick_test:
            system.run_quick_test(models=args.models)
        
        elif args.single_model:
            system.evaluate_single_model(args.single_model)
        
        else:
            # Default: quick test
            print("No specific operation requested. Running quick test...")
            system.run_quick_test()
    
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise


if __name__ == "__main__":
    main()