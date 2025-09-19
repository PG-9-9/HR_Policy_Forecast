# HR Policy Forecast & Immigration Assistant 🇬🇧

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?&logo=docker&logoColor=white)](https://www.docker.com/)

> **Professional UK immigration policy assistance and workforce forecasting system for HR professionals, hiring managers, and employers.**

## 🎯 Overview

The HR Policy Forecast system is an AI-powered application designed to help HR professionals and employers navigate UK immigration requirements while making data-driven workforce planning decisions. It combines advanced time series forecasting with retrieval-augmented generation (RAG) to provide accurate immigration guidance and workforce trend predictions.

### Key Capabilities

- 🔍 **Immigration Policy Guidance**: Real-time advice on UK visa requirements, sponsor licence obligations, and compliance
- 📊 **Workforce Forecasting**: 6-month ahead predictions for UK job vacancy ratios and employment trends  
- 🤖 **Interactive Chat Interface**: Natural language conversations with specialized HR assistant
- 📈 **Advanced Analytics**: Multiple forecasting models (ARIMA, ETS, TFT, Naive) with comprehensive evaluation
- 🔗 **Official Data Integration**: Based on ONS employment statistics and official UK government sources

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │────│   FastAPI App    │────│  ML Forecasting │
│  (Vue.js + UI)  │    │  (Chat + API)    │    │    Pipeline     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                       ┌────────┴────────┐               │
                       │  RAG System     │               │
                       │ (Immigration    │               │
                       │  Knowledge)     │               │
                       └─────────────────┘               │
                                                         │
                       ┌─────────────────┐               │
                       │   Data Layer    │───────────────┘
                       │ (ONS, Models,   │
                       │  Evaluation)    │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker** (optional, for containerized deployment)
- **OpenAI API Key** (for LLM functionality)

### 1. Local Installation

```bash
# Clone the repository
git clone https://github.com/PG-9-9/HR_Policy_Forecast.git
cd HR_Policy_Forecast

# Create and activate conda environment
conda env create -f environment.yml
conda activate mm-hr-policy-forecast

# Or use pip
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 2. Run the Application

```bash
# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open your browser and navigate to:
# http://localhost:8000
```

### 3. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t hr-assistant .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key hr-assistant
```

## 🎯 Core Features

### 💬 HR Assistant Chat Interface

Interactive conversational AI specialized in UK immigration policy:

- **Visa Requirements**: Skilled Worker visas, sponsor licence obligations
- **Compliance Guidance**: Salary thresholds, immigration rule changes
- **Policy Updates**: Real-time information from official UK government sources
- **Hiring Support**: Immigration considerations for international recruitment

**Example Usage:**
```
User: "What are the salary thresholds for Skilled Worker visas in 2024?"
Assistant: "For Skilled Worker visas in 2024, the general salary threshold is £26,200 or the 'going rate' for the specific occupation, whichever is higher..."
```

### 📊 Workforce Forecasting

Advanced time series forecasting for workforce planning:

```python
# Generate 6-month workforce forecast
from forecasting.infer import forecast_months

forecast_df, events = forecast_months(h=6)
print(forecast_df)
```

**Available Models:**
- **Naive Models**: Simple baseline forecasts
- **ARIMA**: Auto-regressive integrated moving average
- **ETS**: Exponential smoothing state space
- **TFT**: Temporal Fusion Transformer (deep learning)
- **Linear Trend**: Trend-based extrapolation

### 🔍 Model Evaluation & Comparison

Professional forecasting evaluation framework:

```python
# Run comprehensive model comparison
python main.py --compare-models

# Quick test with basic models
python main.py --quick-test

# Evaluate specific models
python main.py --models naive arima ets --output results/
```

**Evaluation Metrics:**
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)
- **MASE** (Mean Absolute Scaled Error)

### 🛠️ API Endpoints

#### Chat API
```http
POST /chat
Content-Type: application/json

{
  "session_id": "unique_session_id",
  "message": "What are the requirements for a Skilled Worker visa?",
  "top_k": 6,
  "reset": false
}
```

#### Forecasting API
```http
GET /forecast?h=6

Response:
{
  "horizon": 6,
  "forecast": [
    {"date": "2024-01-01", "prediction": 2.3},
    {"date": "2024-02-01", "prediction": 2.4}
  ],
  "events": []
}
```

#### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "service": "HR Policy Forecast API"
}
```

## 📁 Project Structure

```
mm-hr-policy-forecast/
├── app/                          # FastAPI web application
│   ├── main.py                   # API routes and server configuration
│   ├── db.py                     # Database operations for chat history
│   └── llm.py                    # OpenAI integration and system prompts
├── forecasting/                  # Time series forecasting modules
│   ├── infer.py                  # Inference and prediction functions
│   ├── ets_pipeline.py           # ETS model implementation
│   └── ets_compare.py            # Model comparison utilities
├── models/                       # Forecasting model implementations
│   ├── base_model.py             # Abstract base class for all models
│   ├── arima_model.py            # ARIMA implementation
│   ├── ets_model.py              # Exponential smoothing models
│   ├── naive_models.py           # Baseline naive forecasts
│   └── tft/                      # Temporal Fusion Transformer
├── evaluation/                   # Model evaluation framework
│   ├── evaluator.py              # Professional model evaluator
│   ├── metrics.py                # Evaluation metrics calculation
│   └── cross_validation.py       # Time series cross-validation
├── rag/                          # Retrieval-Augmented Generation
│   ├── build_index.py            # Build search indices for documents
│   ├── retrieve.py               # Document retrieval and search
│   ├── query.py                  # Query processing
│   └── extract_events.py         # Event extraction from documents
├── results/                      # Output directory for forecasts and evaluations
│   ├── export.py                 # Results export utilities
│   ├── visualizations.py         # Chart generation and visualization
│   └── report_generator.py       # Automated report generation
├── data/                         # Data directory
│   ├── raw/                      # Raw ONS and government data
│   ├── processed/                # Cleaned and processed datasets
│   └── external/                 # External datasets (UK corpus, etc.)
├── templates/                    # HTML templates for web interface
│   └── index.html                # Vue.js-based chat interface
├── static/                       # Static web assets
├── docker/                       # Docker configuration files
├── main.py                       # CLI entry point for model evaluation
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment specification
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # Multi-container deployment
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
PORT=8000                         # Server port (default: 8000)
PYTHONUNBUFFERED=1               # Python logging
```

### Model Configuration

Models can be configured in `main.py`:

```python
# Custom model parameters
model_configs = {
    'arima': {'auto_order': True, 'seasonal': True},
    'ets': {'auto_config': True, 'damped': True},
    'tft': {'hidden_size': 64, 'attention_heads': 4}
}

# Run evaluation with custom config
result = evaluator.evaluate_all_models(model_configs=model_configs)
```

## 📊 Data Sources

### Official UK Government Data
- **ONS Employment Statistics**: Job vacancy ratios, unemployment rates
- **Immigration Rules**: Official UK visa requirements and policy updates
- **Government Publications**: Policy changes, salary thresholds, compliance guides

### Supported Data Formats
- **CSV**: Time series data with date and value columns
- **JSON**: Structured policy documents and metadata
- **Text**: Government publications and policy documents

## 🎯 Use Cases

### For HR Professionals
- **International Recruitment**: Understand visa requirements for global talent
- **Compliance Management**: Stay updated on immigration rule changes
- **Workforce Planning**: Forecast labor market trends for strategic hiring
- **Policy Impact**: Assess how immigration changes affect recruitment strategies

### For Hiring Managers
- **Candidate Assessment**: Visa eligibility screening for international applicants
- **Timeline Planning**: Understand processing times for work permits
- **Budget Forecasting**: Predict workforce costs with immigration considerations
- **Risk Management**: Compliance risk assessment for international hires

### For Employers
- **Sponsor Licence Management**: Guidance on sponsor licence obligations
- **Strategic Planning**: Long-term workforce forecasting with immigration factors
- **Compliance Monitoring**: Automated alerts for policy changes
- **Cost Optimization**: Efficient immigration process management

## 🧪 Testing & Evaluation

### Run Model Evaluation
```bash
# Comprehensive evaluation of all models
python main.py --compare-models

# Quick test with subset of models
python main.py --quick-test

# Single model evaluation
python main.py --single-model arima

# Custom evaluation with specific models
python main.py --models naive arima ets tft --output ./results
```

### Example Output
```
================================================================================
 FORECASTING MODEL COMPARISON RESULTS
================================================================================

Models evaluated: 4
Test period: 6 periods
Evaluation time: 15.43 seconds
Best model: ets

Detailed Results:
   model     mae    rmse   smape    mase
0   naive   0.145   0.168   6.23%   1.000
1   arima   0.127   0.149   5.47%   0.875
2     ets   0.108   0.134   4.68%   0.744
3     tft   0.132   0.156   5.71%   0.910

RECOMMENDED MODEL: ets
- sMAPE: 4.68%
- RMSE: 0.134
- MASE: 0.744
```

### Testing the Web Interface
```bash
# Test the chat functionality
curl -X POST "http://localhost:8000/get" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "msg=What are the current UK visa requirements?"

# Test the forecasting API
curl "http://localhost:8000/forecast?h=6"

# Health check
curl "http://localhost:8000/health"
```

## 🚢 Deployment

### AWS Deployment (Recommended)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed AWS deployment instructions including:

- **AWS App Runner**: Easiest deployment option
- **ECS Fargate**: Production-ready container orchestration  
- **EKS**: Advanced Kubernetes deployment
- **Cost optimization** and **security best practices**

### Docker Deployment

```bash
# Build the container
docker build -t hr-assistant .

# Run with environment variables
docker run -d \
  --name hr-assistant \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  hr-assistant

# Or use Docker Compose
docker-compose up -d
```

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest

# Run linting
flake8 . --max-line-length=88
black . --check
```

## 📈 Performance & Scaling

### Forecasting Performance
- **Model Training**: < 5 seconds for most models
- **Inference**: < 1 second for 6-month forecasts
- **Memory Usage**: < 500MB for standard datasets
- **Concurrent Users**: Supports 100+ simultaneous chat sessions

### Optimization Tips
- Use **Docker multi-stage builds** for smaller container images
- Enable **FastAPI caching** for frequently accessed endpoints
- Implement **model result caching** for repeated forecasts
- Use **async endpoints** for better concurrency

## 🔐 Security & Compliance

### Data Protection
- **No PII Storage**: Chat conversations stored locally only
- **API Key Security**: Environment variable management
- **HTTPS Encryption**: SSL/TLS in production deployments
- **Input Validation**: Comprehensive request sanitization

### Immigration Compliance
- **Official Sources Only**: All guidance based on gov.uk documentation
- **Regular Updates**: Automated policy change detection
- **Audit Trail**: Complete logging of all immigration advice provided
- **Disclaimer**: Always recommend consulting immigration specialists for complex cases

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UK Government**: Official immigration policy documentation
- **ONS**: Office for National Statistics employment data
- **OpenAI**: GPT models for natural language processing
- **Vue.js Community**: Frontend framework and components
- **PyTorch Forecasting**: Advanced time series modeling capabilities

## 📞 Support

### Documentation
- **API Documentation**: Available at `/docs` when running the server
- **Model Evaluation Guide**: See `main.py --help` for CLI options
- **Deployment Guide**: Detailed instructions in [DEPLOYMENT.md](DEPLOYMENT.md)

### Getting Help
- **GitHub Issues**: [Report bugs or request features](https://github.com/PG-9-9/HR_Policy_Forecast/issues)
- **Email Support**: Contact the development team
- **Documentation**: Comprehensive guides in the `/docs` directory

### Performance Monitoring
- **Health Endpoint**: `/health` for service monitoring
- **Metrics**: Built-in evaluation metrics for model performance
- **Logging**: Comprehensive application logging for debugging

---


*Last updated: September 2025*
