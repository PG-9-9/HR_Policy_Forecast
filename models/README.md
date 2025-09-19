# ML Forecasting Models - Technical Documentation

This document provides comprehensive technical details about the machine learning forecasting models implemented in the HR Policy Forecast system.

## Quick Navigation
- [Main Project README](../README.md) - Project overview and Docker deployment
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Production Usage](#production-usage)

---

## Model Architecture & Technical Implementation

### Model Selection & Performance Analysis

The forecasting system implements multiple time-series models with comprehensive evaluation:

#### **1. Primary Models Implemented**

**ARIMA (AutoRegressive Integrated Moving Average) - Selected Model:**
- **Rationale**: Chosen due to limited dataset size (~240 monthly observations)
- **Implementation**: Auto-parameter selection with AIC optimization
- **Configuration**: Auto-selects (p,d,q) parameters with max values (3,2,3)
- **Performance**: Best performer for limited data with ~0.15 sMAPE
- **Stationarity**: Automatic ADF testing and differencing

**ETS (Exponential Smoothing) - Secondary Model:**
- **Implementation**: Professional ETS with auto-configuration
- **Performance**: Competitive baseline with seasonal pattern detection
- **Use Case**: Fallback model for rapid deployment

**TFT (Temporal Fusion Transformer) - Deep Learning:**
- **Implementation**: PyTorch Lightning + PyTorch Forecasting
- **Performance Issues**: Poor convergence with limited data
- **Root Cause**: TFT requires 1000+ observations; dataset has ~240
- **Decision**: Excluded from production due to overfitting and instability

#### **2. Model Training Syntax & Commands**

**Train All Models (Comparison):**
```bash
# Run complete model comparison
python forecasting/ets_compare.py

# Output: ETS-focused comparison with ranking
# Saves: data/processed/model_comparison/ets_recommendation.txt
```

**Train Individual Models:**
```bash
# ETS Model Training
python forecasting/ets_pipeline.py
# Output: ETS model with 6-period forecasts and visualizations

# ARIMA Model Training (via evaluation framework)
python -c "
from models import ArimaModel
from evaluation import ModelEvaluator
evaluator = ModelEvaluator('data/processed/models/training_frame.csv')
arima_results = evaluator.evaluate_single_model('arima', display=True)
print('ARIMA Results:', arima_results)
"

# TFT Training (Not recommended for production)
python models/tft/train_tft.py
# Warning: Requires GPU, tends to overfit with limited data
```

**Forecast Generation:**
```bash
# Generate production forecasts (uses best model)
python forecasting/infer.py

# API forecasting (local mode only)
curl "http://localhost:8000/forecast?h=6"
```

#### **3. Data Challenges & Solutions**

**Limited Dataset Problem:**
- **Challenge**: Only ~240 monthly observations from ONS
- **Impact**: Insufficient for deep learning models (TFT)
- **Solution**: Focus on statistical models (ARIMA/ETS) optimized for small datasets

**Feature Engineering:**
- **Policy Events**: Extracted from RAG system using LLM
- **Event Features**: Binary flags and impact scores
- **Temporal Features**: Trend, seasonality, and relative time indices

#### **4. Model Evaluation Framework**

**Metrics Used:**
- **sMAPE**: Primary metric (scale-independent)
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **AIC/BIC**: Information criteria for model selection

**Cross-Validation:**
- **Time Series Split**: 6-month holdout test set
- **Rolling Window**: Expanding window validation
- **Statistical Tests**: Ljung-Box for residual analysis

#### **5. ARIMA Model Selection Rationale**

**Why ARIMA Over TFT:**
```python
# Data Size Analysis
dataset_size = 240  # Monthly observations
tft_minimum = 1000  # Recommended minimum for TFT
data_ratio = dataset_size / tft_minimum  # 0.24 - Insufficient

# Model Performance Comparison
arima_smape = 0.147    # Best performer
ets_smape = 0.152      # Close second
tft_smape = 0.285      # Poor due to overfitting
```

**ARIMA Advantages:**
- **Small Data Optimized**: Designed for limited observations
- **Statistical Foundation**: Established time-series theory
- **Interpretability**: Clear parameter interpretation
- **Stability**: Consistent performance across different time periods

#### **6. TFT Training Issues & Debugging**

**Common TFT Problems Encountered:**
```bash
# Issue 1: Insufficient Data
Error: "Training dataset too small for TFT"
Solution: "Use ARIMA/ETS for datasets < 500 observations"

# Issue 2: Overfitting
Symptom: "High training accuracy, poor validation"
Root Cause: "240 observations vs 32 attention heads"

# Issue 3: Convergence Problems
Lightning Warning: "Training loss plateau at epoch 3"
Cause: "Limited data variability"
```

**TFT Training Command (Educational Only):**
```bash
# Full TFT training (not recommended for production)
cd models/tft/
python train_tft.py

# Expected Output:
# - Lightning logs in models/tft/lightning_logs/
# - Checkpoints in models/tft/checkpoints/
# - Warning: Poor performance due to data limitations
```

#### **7. Explainability & Model Interpretation**

**ARIMA Interpretation:**
- **AR Terms (p)**: How many past values influence prediction
- **Differencing (d)**: Level of trend removal for stationarity
- **MA Terms (q)**: How many past forecast errors are considered

**Feature Importance (Policy Events):**
```python
# Event Impact Analysis
policy_events = {
    "visa_threshold_changes": 0.85,      # High impact
    "sponsor_licence_updates": 0.72,     # Medium-high impact
    "enforcement_changes": 0.43          # Medium impact
}
```

**Note on SHAP Values:**
- **Limitation**: SHAP not implemented for ARIMA in current version
- **Alternative**: Use ARIMA parameter interpretation and residual analysis
- **Future Enhancement**: SHAP integration planned for ensemble models

#### **8. Production Deployment Strategy**

**Model Hierarchy:**
1. **Primary**: ARIMA (auto-parameter selection)
2. **Fallback**: ETS (if ARIMA fails)
3. **Baseline**: Naive forecast (always available)

**Monitoring & Retraining:**
```bash
# Monthly model retraining pipeline
python forecasting/retrain_pipeline.py --schedule monthly

# Performance monitoring
python evaluation/monitor_forecast_accuracy.py --alert-threshold 0.20
```

---

## Training & Evaluation

### Step-by-Step Model Training

#### 1. Data Preparation
```bash
# Download and prepare ONS data
python scripts/download_uk_data.py

# Verify data structure
python -c "
import pandas as pd
df = pd.read_csv('data/raw/ons_vacancies_ratio_total.csv', skiprows=8, header=None)
print(f'Data shape: {df.shape}')
print(f'Date range: {df.iloc[0,0]} to {df.iloc[-1,0]}')
"
```

#### 2. Model Comparison
```bash
# Run comprehensive model comparison
python forecasting/ets_compare.py

# Expected output:
# ┌─────────────┬────────┬────────┬────────┬─────────┐
# │ model_name  │ smape  │ rmse   │ mae    │ r2      │
# ├─────────────┼────────┼────────┼────────┼─────────┤
# │ arima       │ 0.147  │ 0.089  │ 0.071  │ 0.823   │
# │ ets         │ 0.152  │ 0.094  │ 0.075  │ 0.807   │
# │ naive       │ 0.234  │ 0.142  │ 0.118  │ 0.543   │
# └─────────────┴────────┴────────┴────────┴─────────┘
```

#### 3. Individual Model Training
```bash
# Train ETS with full pipeline
python forecasting/ets_pipeline.py

# Output files:
# - data/processed/forecasting_results/ets_forecast_comparison.png
# - data/processed/forecasting_results/ets_forecast_report.txt
```

### Model Files Structure
```
models/
├── __init__.py                    # Model imports
├── base_model.py                  # Abstract base class
├── arima_model.py                 # ARIMA implementation
├── ets_model.py                   # ETS implementation
├── naive_models.py                # Baseline models
├── tft/                           # TFT implementation
│   ├── train_tft.py              # TFT training script
│   ├── tft_model.py              # TFT model class
│   ├── infer.py                  # TFT inference
│   ├── checkpoints/              # Model checkpoints
│   └── lightning_logs/           # Training logs
└── README.md                     # This file
```

---

## Production Usage

### Forecasting API Integration

The forecasting models are integrated into the FastAPI application for production use:

```python
# In app/main.py (when enabled)
from forecasting.infer import forecast_months

@app.get("/forecast", response_model=ForecastResponse)
def forecast_api(h: int = 6):
    """Generate workforce vacancy forecasts"""
    out, events = forecast_months(h)
    return {
        "horizon": h, 
        "forecast": out.to_dict(orient="records"), 
        "events": events
    }
```

### Enabling Forecasting in Local Mode

To enable forecasting in local development:

1. **Install full requirements:**
   ```bash
   pip install -r requirements.txt  # Not requirements-optimized.txt
   ```

2. **Uncomment forecast endpoint in app/main.py:**
   ```python
   # Uncomment lines 91-94:
   @app.get("/forecast", response_model=ForecastResponse)
   def forecast_api(h: int = 6):
       out, events = forecast_months(h)
       return {"horizon": h, "forecast": out.to_dict(orient="records"), "events": events}
   ```

3. **Test the endpoint:**
   ```bash
   uvicorn app.main:app --reload
   curl "http://localhost:8000/forecast?h=6"
   ```

### Model Performance Monitoring

```bash
# Check model accuracy over time
python -c "
from evaluation import ModelEvaluator
evaluator = ModelEvaluator('data/processed/models/training_frame.csv')
results = evaluator.quick_comparison(['arima', 'ets'], display=True)
print('Current best model:', results.iloc[0]['model_name'])
"

# Generate forecast report
python forecasting/ets_pipeline.py
# Check: data/processed/forecasting_results/ets_forecast_report.txt
```

---

## Troubleshooting

### Common Issues

**1. TFT Training Fails:**
```bash
# Error: CUDA out of memory
Solution: Use CPU training or reduce batch_size in train_tft.py

# Error: Dataset too small
Solution: Use ARIMA/ETS instead - TFT needs 1000+ observations
```

**2. ARIMA Convergence Issues:**
```bash
# Warning: Non-stationary data
Solution: Increase max_d parameter for more differencing

# Error: Optimization failed
Solution: Try simpler order like (1,1,1) or use ETS fallback
```

**3. Missing Dependencies:**
```bash
# statsmodels not found
pip install statsmodels

# pytorch-forecasting not found (for TFT)
pip install pytorch-forecasting torch pytorch-lightning
```

### Performance Tuning

**For ARIMA:**
- Increase `max_p`, `max_q` for more complex patterns
- Adjust `max_d` for trend handling
- Use `auto_order=True` for optimization

**For ETS:**
- Enable seasonal components for monthly data
- Adjust smoothing parameters manually if needed
- Use `auto_config=True` for automatic optimization

---

## Future Enhancements

1. **SHAP Integration**: Add explainability for ensemble models
2. **Real-time Updates**: Automatic model retraining pipeline
3. **External Features**: Incorporate economic indicators
4. **Ensemble Methods**: Combine ARIMA + ETS predictions
5. **Uncertainty Quantification**: Enhanced confidence intervals

---

**Navigation:**
- [← Back to Main README](../README.md)
- [Project Structure](../README.md#project-structure)
- [API Documentation](../README.md#api-endpoints)