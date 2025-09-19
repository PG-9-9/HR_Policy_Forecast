@echo off
REM Simplified environment setup for policy_api (Windows)
REM This script creates a clean environment with only chatbot dependencies

echo 🚀 Setting up policy_api environment for chatbot deployment...

REM Check if conda is available
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda not found! Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

REM Remove existing environment if it exists (ignore error if not found)
echo 📦 Checking for existing policy_api environment...
conda env remove -n policy_api -y 2>nul || echo 💡 No existing policy_api environment found

REM Create new environment with Python 3.10
echo 🐍 Creating new conda environment: policy_api
conda create -n policy_api python=3.10 pip -y
if %errorlevel% neq 0 (
    echo ❌ Failed to create conda environment
    pause
    exit /b 1
)

echo ✅ Environment created successfully!
echo.
echo 📦 Installing packages using conda run (this avoids activation issues)...

REM Install packages using conda run to avoid activation issues
echo Installing core web framework...
conda run -n policy_api pip install fastapi==0.116.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install fastapi
    pause
    exit /b 1
)

conda run -n policy_api pip install "uvicorn[standard]==0.35.0"
if %errorlevel% neq 0 (
    echo ❌ Failed to install uvicorn
    pause
    exit /b 1
)

conda run -n policy_api pip install python-multipart==0.0.20
if %errorlevel% neq 0 (
    echo ❌ Failed to install python-multipart
    pause
    exit /b 1
)

echo Installing AI/ML packages...
conda run -n policy_api pip install openai==1.108.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install openai
    pause
    exit /b 1
)

conda run -n policy_api pip install jinja2==3.1.6
if %errorlevel% neq 0 (
    echo ❌ Failed to install jinja2
    pause
    exit /b 1
)

conda run -n policy_api pip install python-dotenv==1.1.1
if %errorlevel% neq 0 (
    echo ❌ Failed to install python-dotenv
    pause
    exit /b 1
)

echo Installing data processing packages...
conda run -n policy_api pip install pandas==2.3.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install pandas
    pause
    exit /b 1
)

conda run -n policy_api pip install numpy==2.1.3
if %errorlevel% neq 0 (
    echo ❌ Failed to install numpy
    pause
    exit /b 1
)

conda run -n policy_api pip install requests==2.32.3
if %errorlevel% neq 0 (
    echo ❌ Failed to install requests
    pause
    exit /b 1
)

echo Installing RAG/Search packages...
conda run -n policy_api pip install sentence-transformers==5.1.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install sentence-transformers
    pause
    exit /b 1
)

conda run -n policy_api pip install faiss-cpu==1.12.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install faiss-cpu
    pause
    exit /b 1
)

conda run -n policy_api pip install rank-bm25==0.2.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install rank-bm25
    pause
    exit /b 1
)

echo Installing document parsing packages...
conda run -n policy_api pip install beautifulsoup4==4.12.3
if %errorlevel% neq 0 (
    echo ❌ Failed to install beautifulsoup4
    pause
    exit /b 1
)

conda run -n policy_api pip install pdfminer.six
if %errorlevel% neq 0 (
    echo ❌ Failed to install pdfminer.six
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!

echo 🧪 Testing imports...
conda run -n policy_api python -c "
import fastapi
import uvicorn
import openai
import jinja2
import pandas
import numpy
import sentence_transformers
import faiss
import rank_bm25
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
print('✅ All imports successful!')
"

if %errorlevel% neq 0 (
    echo ❌ Import test failed!
    pause
    exit /b 1
)

echo 🎉 Environment setup complete!
echo.
echo 📝 To use this environment:
echo    conda activate policy_api
echo    cd f:\Projects\MM\mm-hr-policy-forecast
echo    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 💡 Environment is ready for chatbot deployment!
echo.
pause