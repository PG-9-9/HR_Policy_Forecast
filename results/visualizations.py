"""
Professional Chart Generation Module
High-quality visualizations for model comparison and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
sns.set_palette("husl")

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


@dataclass
class PlotConfig:
    """Configuration for plot styling and layout."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = 'professional'
    color_palette: str = 'husl'
    font_size: int = 10
    title_size: int = 14
    save_format: str = 'png'
    transparent: bool = False


class ChartGenerator:
    """
    Professional chart generator for forecasting model analysis.
    
    Creates publication-quality visualizations with consistent styling.
    """
    
    def __init__(self, config: PlotConfig = None):
        """Initialize chart generator with configuration."""
        self.config = config or PlotConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib styling for professional appearance."""
        plt.rcParams.update({
            'figure.figsize': self.config.figsize,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size - 1,
            'ytick.labelsize': self.config.font_size - 1,
            'legend.fontsize': self.config.font_size - 1,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.edgecolor': 'gray',
            'axes.linewidth': 0.8
        })
    
    def create_model_comparison_chart(self,
                                    comparison_df: pd.DataFrame,
                                    primary_metric: str = 'smape',
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create professional model comparison chart.
        
        Args:
            comparison_df: DataFrame with model comparison results
            primary_metric: Primary metric to highlight
            save_path: Optional path to save chart
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Primary metric comparison (bar chart)
        models = comparison_df['model_name']
        primary_values = comparison_df[primary_metric]
        
        bars = ax1.bar(models, primary_values, 
                      color=['#1f77b4' if i > 0 else '#d62728' for i in range(len(models))],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Highlight best model
        if len(bars) > 0:
            bars[0].set_color('#2ca02c')
            bars[0].set_alpha(1.0)
        
        ax1.set_title(f'{primary_metric.upper()} Comparison (Lower is Better)', fontweight='bold')
        ax1.set_ylabel(primary_metric.upper())
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, primary_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Multiple metrics radar chart (simplified as grouped bar chart)
        metrics_to_plot = ['mae', 'rmse', 'smape', 'mase']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if len(available_metrics) > 1:
            # Normalize metrics for comparison (0-1 scale)
            normalized_data = comparison_df[available_metrics].copy()
            for col in available_metrics:
                max_val = normalized_data[col].max()
                min_val = normalized_data[col].min()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
            
            x_pos = np.arange(len(models))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                offset = (i - len(available_metrics)/2 + 0.5) * width
                ax2.bar(x_pos + offset, normalized_data[metric], width, 
                       label=metric.upper(), alpha=0.8)
            
            ax2.set_title('Normalized Metrics Comparison', fontweight='bold')
            ax2.set_ylabel('Normalized Score (0-1)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(models, rotation=45)
            ax2.legend()
        
        # 3. Model ranking
        ranks = comparison_df['rank'] if 'rank' in comparison_df.columns else range(1, len(models) + 1)
        colors = ['#2ca02c', '#ff7f0e', '#8c564b'] + ['#lightgray'] * (len(ranks) - 3)
        
        bars = ax3.barh(models, ranks, color=colors[:len(ranks)], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_title('Model Ranking (1 = Best)', fontweight='bold')
        ax3.set_xlabel('Rank')
        ax3.invert_yaxis()
        
        # Add rank numbers
        for bar, rank in zip(bars, ranks):
            width = bar.get_width()
            ax3.text(width/2, bar.get_y() + bar.get_height()/2,
                    str(int(rank)), ha='center', va='center', fontweight='bold', color='white')
        
        # 4. Performance summary table
        ax4.axis('off')
        
        # Create summary table
        if len(comparison_df) > 0:
            summary_data = []
            for _, row in comparison_df.head(5).iterrows():  # Top 5 models
                summary_data.append([
                    row['model_name'],
                    f"{row[primary_metric]:.4f}",
                    f"{row.get('mae', 0):.4f}",
                    f"{row.get('rmse', 0):.4f}"
                ])
            
            table = ax4.table(cellText=summary_data,
                            colLabels=['Model', primary_metric.upper(), 'MAE', 'RMSE'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.1, 0.8, 0.8])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(summary_data) + 1):
                for j in range(4):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    elif i == 1:  # Best model
                        cell.set_facecolor('#E8F5E8')
                    else:
                        cell.set_facecolor('#F5F5F5')
                    cell.set_edgecolor('white')
        
        ax4.set_title('Top Models Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_forecast_comparison_chart(self,
                                       actual_data: pd.Series,
                                       forecasts: Dict[str, np.ndarray],
                                       dates: Optional[pd.DatetimeIndex] = None,
                                       confidence_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create forecast comparison chart showing actual vs predicted values.
        
        Args:
            actual_data: Actual time series values
            forecasts: Dictionary of model_name -> predictions
            dates: Optional dates for x-axis
            confidence_intervals: Optional confidence intervals per model
            save_path: Optional path to save chart
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Forecast Comparison: Actual vs Predicted', fontsize=16, fontweight='bold')
        
        # Prepare x-axis
        if dates is not None:
            x_actual = dates[:len(actual_data)]
            x_forecast = dates[len(actual_data)-len(list(forecasts.values())[0]):]
        else:
            x_actual = range(len(actual_data))
            x_forecast = range(len(actual_data)-len(list(forecasts.values())[0]), len(actual_data))
        
        # 1. Time series plot
        ax1.plot(x_actual, actual_data, 'ko-', label='Actual', linewidth=2, markersize=4)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
        
        for i, (model_name, predictions) in enumerate(forecasts.items()):
            color = colors[i]
            ax1.plot(x_forecast, predictions, 'o--', color=color, 
                    label=f'{model_name}', alpha=0.8, linewidth=2, markersize=3)
            
            # Add confidence intervals if available
            if confidence_intervals and model_name in confidence_intervals:
                lower, upper = confidence_intervals[model_name]
                ax1.fill_between(x_forecast, lower, upper, alpha=0.2, color=color)
        
        ax1.set_title('Time Series Forecast Comparison', fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        if dates is not None:
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Residuals plot
        for i, (model_name, predictions) in enumerate(forecasts.items()):
            residuals = actual_data.iloc[-len(predictions):].values - predictions
            color = colors[i]
            ax2.plot(x_forecast, residuals, 'o-', color=color, label=f'{model_name}', alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Forecast Residuals', fontweight='bold')
        ax2.set_ylabel('Residual (Actual - Predicted)')
        ax2.set_xlabel('Time')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        if dates is not None:
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_metrics_heatmap(self,
                             comparison_df: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap of model performance metrics.
        
        Args:
            comparison_df: DataFrame with model comparison results
            save_path: Optional path to save chart
            
        Returns:
            Matplotlib figure object
        """
        # Select numeric columns for heatmap
        numeric_cols = ['mae', 'rmse', 'smape', 'mase']
        available_cols = [col for col in numeric_cols if col in comparison_df.columns]
        
        if not available_cols:
            raise ValueError("No numeric metrics available for heatmap")
        
        # Prepare data
        heatmap_data = comparison_df.set_index('model_name')[available_cols]
        
        # Normalize data for better visualization (rank-based)
        normalized_data = heatmap_data.rank(method='min')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        sns.heatmap(normalized_data, 
                   annot=heatmap_data.round(4),  # Show actual values
                   fmt='',
                   cmap='RdYlGn_r',  # Red (bad) to Green (good)
                   center=len(comparison_df)/2,
                   cbar_kws={'label': 'Rank (Lower is Better)'},
                   ax=ax)
        
        ax.set_title('Model Performance Heatmap\n(Lower Values = Better Performance)', 
                    fontweight='bold', pad=20)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Models')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_model_diagnostics_chart(self,
                                     model_metrics: List,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive model diagnostics chart.
        
        Args:
            model_metrics: List of ModelMetrics objects
            save_path: Optional path to save chart
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Diagnostics Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data
        models = [m.model_name for m in model_metrics]
        mae_values = [m.mae for m in model_metrics]
        rmse_values = [m.rmse for m in model_metrics]
        smape_values = [m.smape for m in model_metrics]
        mase_values = [m.mase for m in model_metrics]
        
        # 1. MAE vs RMSE scatter plot
        ax1.scatter(mae_values, rmse_values, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (mae_values[i], rmse_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('MAE')
        ax1.set_ylabel('RMSE')
        ax1.set_title('MAE vs RMSE', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. SMAPE distribution
        ax2.bar(models, smape_values, alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.set_title('SMAPE by Model', fontweight='bold')
        ax2.set_ylabel('SMAPE')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. MASE comparison
        colors = ['green' if x < 1 else 'red' for x in mase_values]
        ax3.bar(models, mase_values, alpha=0.7, color=colors, edgecolor='black')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.8, label='MASE = 1 (baseline)')
        ax3.set_title('MASE by Model\n(< 1 = Better than naive)', fontweight='bold')
        ax3.set_ylabel('MASE')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # 4. Overall performance ranking
        # Create a composite score (normalized and averaged)
        scores = []
        for m in model_metrics:
            # Normalize each metric (lower is better)
            norm_mae = m.mae / max(mae_values) if max(mae_values) > 0 else 0
            norm_rmse = m.rmse / max(rmse_values) if max(rmse_values) > 0 else 0
            norm_smape = m.smape / max(smape_values) if max(smape_values) > 0 else 0
            norm_mase = m.mase / max(mase_values) if max(mase_values) > 0 else 0
            
            composite_score = (norm_mae + norm_rmse + norm_smape + norm_mase) / 4
            scores.append(composite_score)
        
        # Sort by composite score
        sorted_indices = np.argsort(scores)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        colors = ['#2ca02c', '#ff7f0e', '#d62728'] + ['lightgray'] * (len(sorted_models) - 3)
        ax4.barh(sorted_models, sorted_scores, color=colors[:len(sorted_models)], alpha=0.8)
        ax4.set_title('Composite Performance Score\n(Lower is Better)', fontweight='bold')
        ax4.set_xlabel('Normalized Composite Score')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, save_path: str):
        """Save figure with consistent settings."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, 
                   dpi=self.config.dpi,
                   bbox_inches='tight',
                   transparent=self.config.transparent,
                   format=self.config.save_format)
        
        print(f"Chart saved to: {save_path}")


def create_summary_dashboard(comparison_df: pd.DataFrame, 
                           actual_data: pd.Series,
                           forecasts: Dict[str, np.ndarray],
                           output_dir: str = "results/charts") -> Dict[str, str]:
    """
    Create a complete dashboard of charts for model comparison.
    
    Args:
        comparison_df: Model comparison DataFrame
        actual_data: Actual time series data
        forecasts: Dictionary of model forecasts
        output_dir: Directory to save charts
        
    Returns:
        Dictionary of chart names and their file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator = ChartGenerator()
    saved_charts = {}
    
    # 1. Main comparison chart
    fig1 = generator.create_model_comparison_chart(
        comparison_df, 
        save_path=output_path / "model_comparison.png"
    )
    saved_charts['model_comparison'] = str(output_path / "model_comparison.png")
    plt.close(fig1)
    
    # 2. Forecast comparison
    fig2 = generator.create_forecast_comparison_chart(
        actual_data,
        forecasts,
        save_path=output_path / "forecast_comparison.png"
    )
    saved_charts['forecast_comparison'] = str(output_path / "forecast_comparison.png")
    plt.close(fig2)
    
    # 3. Metrics heatmap
    try:
        fig3 = generator.create_metrics_heatmap(
            comparison_df,
            save_path=output_path / "metrics_heatmap.png"
        )
        saved_charts['metrics_heatmap'] = str(output_path / "metrics_heatmap.png")
        plt.close(fig3)
    except Exception as e:
        print(f"Could not create heatmap: {e}")
    
    return saved_charts