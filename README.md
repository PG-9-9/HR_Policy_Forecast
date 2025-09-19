# HR Policy Forecast & Immigration Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?&logo=docker&logoColor=white)](https://www.docker.com/)

> **(Almost) Production-style UK immigration policy assistant for HR professionals, hiring managers, and employers.**

**Sweet Note2:** Almost Production Ready, Well lot more to test, and debug, haven't worked on adding runner(yet!)
**Sweet Note1:** Start chat by saying Hi

**Live Demo:** [http://project-demo.live](http://project-demo.live)

## Overview

The HR Policy Forecast system is an AI-powered conversational assistant specifically designed to help HR professionals navigate complex immigration requirements in their daily operations. Originally conceptualized for the German/EU market, this system was rapidly prototyped using UK immigration data due to better data availability and time constraints, demonstrating a complete end-to-end solution in under 36 hours of development.

### Target Users: HR Professionals & Immigration Challenges

HR teams face increasing complexity when managing international talent acquisition and compliance. This system addresses common HR pain points:

**Typical HR Questions Answered:**
- "What visa category should we use for a software engineer from India?"
- "What are the minimum salary requirements for sponsoring a candidate?"
- "How long does the sponsor licence application process take?"
- "What documentation do we need for a Skilled Worker visa application?"
- "Can we hire someone on a student visa for a permanent role?"
- "What are our compliance obligations as a sponsor?"
- "How do recent policy changes affect our current applications?"

**HR Professional Benefits:**
- **Instant Policy Guidance**: Get immediate answers to immigration queries without lengthy research
- **Compliance Assurance**: Stay updated on changing immigration rules and requirements
- **Cost Efficiency**: Reduce reliance on expensive immigration lawyers for routine questions
- **Risk Mitigation**: Understand compliance obligations and avoid costly mistakes
- **Strategic Planning**: Make informed decisions about international hiring strategies
- **Time Savings**: Eliminate hours of manual research through gov.uk documents

### Rapid Development & Replication Strategy

**Development Timeline:** Complete system built in under 36 hours (20% code reused from previous deployment projects)

**German/EU Market Adaptation:**
While initially designed for the German market, UK data was used for rapid prototyping. The system can be easily adapted for German immigration law:

1. **Data Sources Replacement:**
   - Replace UK gov.uk documents with German Federal Office for Migration and Refugees (BAMF) publications
   - Integrate EU Blue Card requirements and German skilled worker immigration act documents
   - Add German labor market data from Bundesagentur für Arbeit

2. **Document Processing:**
   - German language document processing (UTF-8 encoding for umlauts)
   - Legal terminology adaptation for German immigration law
   - EU directive compliance documentation

3. **Localization Steps:**
   ```bash
   # Replace data sources in /data/external/
   # German immigration documents instead of UK corpus
   # Update RAG index building in rag/build_index.py
   # Modify system prompts in app/llm.py for German context
   ```

4. **Regulatory Adaptation:**
   - German visa categories (EU Blue Card, Skilled Worker, etc.)
   - German salary thresholds and qualification requirements
   - German compliance and reporting obligations

The modular RAG architecture allows for seamless adaptation to any country's immigration system by simply replacing the document corpus and updating the system prompts.

### Key Capabilities

- **Immigration Policy Guidance**: Real-time advice on UK visa requirements, sponsor licence obligations, and compliance
- **Workforce Vacancy Forecasting**: Time-series forecasting of UK job market trends (local deployment only)
- **Interactive Chat Interface**: Natural language conversations with specialized HR assistant
- **Official Data Integration**: Based on ONS employment statistics and official UK government sources
- **Dual Deployment Modes**: Full ML stack locally or minimal containerized deployment
- **Production Deployment**: Live on AWS EC2 with nginx reverse proxy and domain setup

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │────│   FastAPI App    │────│  RAG System     │
│  (Vue.js + UI)  │    │  (Chat + API)    │    │ (Immigration    │
└─────────────────┘    └──────────────────┘    │  Knowledge)     │
                                │                └─────────────────┘
                       ┌────────┴────────┐               
                       │   Data Layer    │        ┌─────────────────┐
                       │ (UK Policy,     │────────│ Forecasting ML  │
                       │  Documents)     │        │ (Local Mode)    │
                       └─────────────────┘        └─────────────────┘
```

### Deployment Modes

**Local Development (Full Stack):**
- Complete ML forecasting capabilities with PyTorch
- Time-series forecasting of UK job market trends
- All features including workforce predictions

**Docker Production (Minimal):**
- Immigration policy assistant only
- Optimized ~4GB image size
- Production-ready with health checks

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker** (for containerized deployment)
- **OpenAI API Key** (for LLM functionality)

### 1. Local Development (Full Stack with Forecasting)

```bash
# Clone the repository
git clone https://github.com/PG-9-9/HR_Policy_Forecast.git
cd HR_Policy_Forecast

# Install full dependencies (includes ML forecasting)
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Enable forecasting endpoint (uncomment in app/main.py)
# Uncomment lines 91-94 in app/main.py:
# @app.get("/forecast", response_model=ForecastResponse)
# def forecast_api(h: int = 6):
#     out, events = forecast_months(h)
#     return {"horizon": h, "forecast": out.to_dict(orient="records"), "events": events}

# Start the FastAPI server with all features
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open your browser and navigate to:
# http://localhost:8000
# Available endpoints: /chat, /forecast, /health
```

**Note:** Local mode includes workforce vacancy forecasting via `/forecast` endpoint when enabled.

**🔧 To enable forecasting:** Uncomment lines 91-94 in `app/main.py` and install full dependencies (`pip install -r requirements.txt`)

### 2. Docker Deployment (Production - Policy Assistant Only)

```bash
# Build optimized Docker image (no forecasting)
docker build -t mm-hr-optimized -f Dockerfile.optimized .

# Run with environment file
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name hr-chatbot \
  --restart unless-stopped \
  mm-hr-optimized
```

**Note:** Docker mode excludes forecasting to minimize image size (~4GB vs ~12GB with ML dependencies).

## Core Features

### HR Assistant Chat Interface

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

### Workforce Vacancy Forecasting (Local Mode Only)

Time-series forecasting capabilities for strategic workforce planning:

- **Job Market Trends**: Predict UK vacancy ratios up to 12 months ahead
- **Event Impact Analysis**: Assess how policy changes affect job markets
- **Strategic Planning**: Data-driven insights for recruitment timelines
- **Historical Analysis**: Trend analysis of UK employment patterns

**Forecasting Use Cases:**
- **Recruitment Planning**: Anticipate tight labor markets for strategic hiring
- **Budget Allocation**: Plan recruitment spend based on market predictions
- **Visa Timing**: Optimize visa application timing with market forecasts
- **Workforce Strategy**: Long-term planning based on employment trends

**Example API Usage:**
```bash
# Get 6-month job market forecast
curl "http://localhost:8000/forecast?h=6"

Response:
{
  "horizon": 6,
  "forecast": [
    {"date": "2025-01-01", "prediction": 2.34},
    {"date": "2025-02-01", "prediction": 2.28},
    ...
  ],
  "events": []
}
```

### Retrieval-Augmented Generation (RAG)

Advanced document retrieval system for accurate immigration guidance:

- **Document Search**: FAISS vector search and BM25 keyword matching
- **Context Relevance**: Automatic relevance filtering for immigration topics
- **Official Sources**: UK government documents and policy publications
- **Real-time Updates**: Dynamic context retrieval for current policies

### Workforce Forecasting (Local Mode)

Time-series forecasting for strategic workforce planning:

- **Primary Model**: ARIMA (selected for limited dataset ~240 observations)
- **Secondary Model**: ETS (Exponential Smoothing fallback)
- **Performance**: ~0.15 sMAPE accuracy on 6-month forecasts
- **Features**: Policy event integration, automated model selection

**Quick Forecasting Commands:**
```bash
# Model comparison and training
python forecasting/ets_compare.py

# Generate forecasts via API (local mode)
curl "http://localhost:8000/forecast?h=6"
```

**📊 For detailed ML implementation, training commands, and technical analysis → [models/README.md](models/README.md)**

## API Endpoints

### Chat API
```http
POST /get
Content-Type: application/x-www-form-urlencoded

msg=What are the requirements for a Skilled Worker visa?
```

### Chat API (JSON)
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

### Workforce Forecasting API (Local Mode Only)
```http
GET /forecast?h=6
Content-Type: application/json

Response:
{
  "horizon": 6,
  "forecast": [
    {"date": "2025-01-01", "prediction": 2.34},
    {"date": "2025-02-01", "prediction": 2.28}
  ],
  "events": []
}
```

### Health Check
```http
GET /health

Response:
{
  "status": "healthy",
  "service": "HR Policy Forecast API"
}
```

## Project Structure

```
mm-hr-policy-forecast/
├── app/                          # FastAPI web application
│   ├── main.py                   # API routes and server configuration
│   ├── db.py                     # Database operations for chat history
│   └── llm.py                    # OpenAI integration and system prompts
├── rag/                          # Retrieval-Augmented Generation system
│   ├── build_index.py            # Build search indices for documents
│   ├── retrieve.py               # Document retrieval and search
│   ├── query.py                  # Query processing
│   └── extract_events.py         # Event extraction from documents
├── forecasting/                  # Time-series forecasting (local mode)
│   ├── infer.py                  # Forecasting inference engine
│   ├── ets_pipeline.py           # ETS forecasting pipeline
│   └── ets_compare.py            # Model comparison framework
├── models/                       # ML model implementations
│   ├── README.md                 # Detailed ML documentation
│   ├── arima_model.py            # ARIMA time-series model
│   ├── ets_model.py              # ETS exponential smoothing
│   ├── naive_models.py           # Baseline models
│   └── tft/                      # TFT deep learning (experimental)
│       ├── train_tft.py          # TFT training script
│       └── tft_model.py          # TFT implementation
├── data/                         # Data directory
│   ├── raw/                      # Raw ONS and government data
│   ├── processed/                # Cleaned and processed datasets
│   └── external/                 # External datasets (UK corpus, etc.)
├── templates/                    # HTML templates for web interface
│   └── index.html                # Vue.js-based chat interface
├── static/                       # Static web assets
├── Dockerfile.optimized          # Production Docker configuration (no ML)
├── requirements.txt              # Full dependencies (with forecasting)
├── requirements-optimized.txt    # Production Python dependencies (minimal)
├── main.py                       # CLI entry point for batch processing
└── README.md                     # This file
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
PORT=8000                         # Server port (default: 8000)
ENVIRONMENT=production            # Environment setting
```

### Docker Configuration

The project uses an optimized Docker setup for production deployment:

- **Multi-stage build** for minimal image size
- **Non-root user** for security
- **Pre-cached models** for faster startup
- **Health checks** for monitoring

## Data Sources

### Official UK Government Data
- **ONS Employment Statistics**: Job vacancy ratios, unemployment rates
- **Immigration Rules**: Official UK visa requirements and policy updates
- **Government Publications**: Policy changes, salary thresholds, compliance guides

### Supported Data Formats
- **CSV**: Time series data with date and value columns
- **JSON**: Structured policy documents and metadata
- **Text**: Government publications and policy documents

## Use Cases

### For HR Professionals
- **International Recruitment**: Understand visa requirements for global talent
- **Compliance Management**: Stay updated on immigration rule changes
- **Policy Impact**: Assess how immigration changes affect recruitment strategies
- **Workforce Forecasting**: Plan recruitment based on predicted job market trends
- **Strategic Timing**: Optimize hiring timelines using vacancy forecasts

### For Hiring Managers
- **Candidate Assessment**: Visa eligibility screening for international applicants
- **Timeline Planning**: Understand processing times for work permits
- **Risk Management**: Compliance risk assessment for international hires
- **Market Intelligence**: Use forecasting to plan recruitment campaigns

### For Employers
- **Sponsor Licence Management**: Guidance on sponsor licence obligations
- **Strategic Planning**: Immigration considerations in workforce planning
- **Compliance Monitoring**: Stay informed on policy changes
- **Cost Optimization**: Efficient immigration process management
- **Budget Planning**: Forecast-driven recruitment budget allocation

### Forecasting-Specific Use Cases (Local Mode)
- **Seasonal Planning**: Predict tight labor markets for key recruitment periods
- **Resource Allocation**: Allocate recruitment resources based on market forecasts
- **Competitive Intelligence**: Understand when to accelerate hiring before market tightens
- **Long-term Strategy**: Multi-month workforce planning based on employment trends

**💡 For detailed model training, ARIMA vs TFT analysis, and technical implementation → [models/README.md](models/README.md)**

## Testing

### Testing the Web Interface
```bash
# Test the chat functionality
curl -X POST "http://localhost:8000/get" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "msg=What are the current UK visa requirements?"

# Test the JSON API
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test", "message": "Hello", "reset": false}'

# Test forecasting (local mode only)
curl "http://localhost:8000/forecast?h=3"

# Health check
curl "http://localhost:8000/health"
```

### Live Demo
Test the live application at: [http://project-demo.live](http://project-demo.live)

## Deployment

### Production Deployment (AWS EC2)

The application is currently deployed on AWS EC2 with the following setup:

- **EC2 Instance**: Ubuntu 22.04 LTS
- **Domain**: project-demo.live
- **Reverse Proxy**: nginx
- **Container**: Docker with optimized image
- **Security**: AWS Security Groups

#### Quick Deployment Commands
```bash
# 1. SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Clone and build
git clone https://github.com/PG-9-9/HR_Policy_Forecast.git
cd HR_Policy_Forecast
docker build -t mm-hr-optimized -f Dockerfile.optimized .

# 3. Setup secure environment
sudo mkdir -p /opt/chatbot
sudo chown ubuntu:ubuntu /opt/chatbot
echo "OPENAI_API_KEY=your-api-key-here" > /opt/chatbot/.env
chmod 600 /opt/chatbot/.env

# 4. Run securely
docker run -d -p 8000:8000 --env-file /opt/chatbot/.env --name hr-chatbot --restart unless-stopped mm-hr-optimized

# 5. Setup nginx (optional)
sudo apt install -y nginx
# Configure nginx for domain and SSL
```

### Local Development

```bash
# Install in development mode (full stack with forecasting)
pip install -r requirements.txt

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Test forecasting endpoint
curl "http://localhost:8000/forecast?h=6"
```

### Docker Development

```bash
# Build development image (minimal stack, no forecasting)
docker build -t hr-assistant-dev -f Dockerfile.optimized .

# Run development container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key hr-assistant-dev
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-optimized.txt

# Run the application locally
uvicorn app.main:app --reload

# Test the application
curl http://localhost:8000/health
```

## Performance & Scaling

### Application Performance
- **Response Time**: < 2 seconds for most queries
- **Memory Usage**: < 1GB for standard operations
- **Concurrent Users**: Supports 50+ simultaneous chat sessions
- **Docker Image**: Optimized to < 4GB

### Optimization Features
- **Optimized Docker build** with multi-stage process
- **Pre-cached models** for faster startup
- **Efficient RAG system** with FAISS and BM25
- **Session management** for chat continuity

## Security & Compliance

### Data Protection
- **No PII Storage**: Chat conversations stored locally only
- **API Key Security**: Environment variable management
- **HTTPS Support**: SSL/TLS configuration available
- **Input Validation**: Comprehensive request sanitization

### Immigration Compliance
- **Official Sources Only**: All guidance based on gov.uk documentation
- **Audit Trail**: Complete logging of all immigration advice provided
- **Disclaimer**: Always recommend consulting immigration specialists for complex cases

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UK Government**: Official immigration policy documentation
- **ONS**: Office for National Statistics employment data
- **OpenAI**: GPT models for natural language processing
- **Vue.js Community**: Frontend framework and components
- **AI-QL/chat-ui**: Single-File AI Chatbot UI with Multimodal & MCP Support[](https://github.com/AI-QL/chat-ui)

## Support

### Getting Help
- **GitHub Issues**: [Report bugs or request features](https://github.com/PG-9-9/HR_Policy_Forecast/issues)
- **Live Demo**: [http://project-demo.live](http://project-demo.live)
- **API Documentation**: Available at `/docs` when running the server

### Performance Monitoring
- **Health Endpoint**: `/health` for service monitoring
- **Logging**: Comprehensive application logging for debugging

---

*Last updated: September 2025*


