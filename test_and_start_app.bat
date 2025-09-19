@echo off
REM Test script for policy_api environment
REM Validates all dependencies and starts the FastAPI app

echo 🧪 Testing policy_api environment and app startup...

REM Check if we're in the right environment
python -c "import sys; print(f'Python: {sys.executable}')"
if %errorlevel% neq 0 (
    echo ❌ Python not available! Make sure policy_api environment is activated
    echo Run: conda activate policy_api
    pause
    exit /b 1
)

echo 📦 Checking core dependencies...

REM Test core web framework
echo Testing FastAPI...
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')"
if %errorlevel% neq 0 (
    echo ❌ FastAPI not available
    echo Installing FastAPI...
    pip install fastapi==0.116.2
)

echo Testing Uvicorn...
python -c "import uvicorn; print(f'✅ Uvicorn {uvicorn.__version__}')"
if %errorlevel% neq 0 (
    echo ❌ Uvicorn not available
    echo Installing Uvicorn...
    pip install "uvicorn[standard]==0.35.0"
)

echo Testing python-multipart...
python -c "import multipart; print('✅ python-multipart available')"
if %errorlevel% neq 0 (
    echo ❌ python-multipart not available
    echo Installing python-multipart...
    pip install python-multipart==0.0.20
)

REM Test AI/ML dependencies
echo Testing OpenAI...
python -c "import openai; print(f'✅ OpenAI {openai.__version__}')"
if %errorlevel% neq 0 (
    echo ❌ OpenAI not available
    echo Installing OpenAI...
    pip install openai==1.108.0
)

echo Testing sentence-transformers...
python -c "import sentence_transformers; print(f'✅ sentence-transformers {sentence_transformers.__version__}')"
if %errorlevel% neq 0 (
    echo ❌ sentence-transformers not available
    echo Installing sentence-transformers...
    pip install sentence-transformers==5.1.0
)

echo Testing FAISS...
python -c "import faiss; print('✅ FAISS available')"
if %errorlevel% neq 0 (
    echo ❌ FAISS not available
    echo Installing FAISS...
    pip install faiss-cpu==1.12.0
)

REM Test other dependencies
echo Testing other dependencies...
python -c "
try:
    import jinja2
    import pandas
    import numpy
    import requests
    from dotenv import load_dotenv
    from bs4 import BeautifulSoup
    from rank_bm25 import BM25Okapi
    print('✅ All dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ Some dependencies are missing
    echo Installing remaining packages...
    pip install jinja2==3.1.6 pandas==2.3.2 numpy==1.26.4 requests==2.32.5 python-dotenv==1.1.1 beautifulsoup4==4.13.5 rank-bm25==0.2.2 pdfminer.six==20250506
)

echo 🔧 Testing app import...
python -c "
try:
    from app.main import app
    print('✅ App imports successfully')
except ImportError as e:
    print(f'❌ App import failed: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ App import failed! Check app/main.py for issues
    pause
    exit /b 1
)

echo 🚀 Starting FastAPI server...
echo 📡 Server will be available at: http://localhost:8000
echo � Note: Use localhost:8000 in browser (NOT 0.0.0.0:8000)
echo �🔄 Press Ctrl+C to stop the server
echo.

REM Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000