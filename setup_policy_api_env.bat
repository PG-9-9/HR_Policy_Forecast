@echo off
REM Automated environment setup for policy_api (Windows)
REM This script creates a clean environment with only chatbot dependencies

echo 🚀 Setting up policy_api environment for chatbot deployment...

REM Remove existing environment if it exists (ignore error if not found)
echo 📦 Checking for existing policy_api environment...
conda remove -n policy_api --all -y 2>nul || echo 💡 No existing policy_api environment found (this is normal)

REM Create new environment with Python 3.10
echo 🐍 Creating new conda environment: policy_api
conda create -n policy_api python=3.10 -y
if %errorlevel% neq 0 (
    echo ❌ Failed to create conda environment
    exit /b 1
)

REM Activate the environment (using conda activate doesn't work in batch files)
echo ⚡ Environment created! Activating for package installation...
call conda activate policy_api 2>nul || (
    echo 💡 Using pip with conda environment path...
    set "CONDA_ENV_PATH=F:\Conda\envs\policy_api"
)

REM Install packages one by one with error checking
echo 📚 Installing core web framework packages...
pip install fastapi==0.116.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install fastapi
    exit /b 1
)

pip install "uvicorn[standard]==0.35.0"
if %errorlevel% neq 0 (
    echo ❌ Failed to install uvicorn
    exit /b 1
)

pip install python-multipart==0.0.20
if %errorlevel% neq 0 (
    echo ❌ Failed to install python-multipart
    exit /b 1
)

echo 🤖 Installing OpenAI and template packages...
pip install openai==1.108.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install openai
    exit /b 1
)

pip install jinja2==3.1.6
if %errorlevel% neq 0 (
    echo ❌ Failed to install jinja2
    exit /b 1
)

pip install python-dotenv==1.1.1
if %errorlevel% neq 0 (
    echo ❌ Failed to install python-dotenv
    exit /b 1
)

echo 📊 Installing data processing packages...
pip install pandas==2.3.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install pandas
    exit /b 1
)

pip install numpy==1.26.4
if %errorlevel% neq 0 (
    echo ❌ Failed to install numpy
    exit /b 1
)

pip install requests==2.32.5
if %errorlevel% neq 0 (
    echo ❌ Failed to install requests
    exit /b 1
)

echo 🔍 Installing RAG/Search packages...
pip install sentence-transformers==5.1.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install sentence-transformers
    exit /b 1
)

pip install faiss-cpu==1.12.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install faiss-cpu
    exit /b 1
)

pip install rank-bm25==0.2.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install rank-bm25
    exit /b 1
)

echo 📄 Installing document parsing packages...
pip install beautifulsoup4==4.13.5
if %errorlevel% neq 0 (
    echo ❌ Failed to install beautifulsoup4
    exit /b 1
)

pip install pdfminer.six==20250506
if %errorlevel% neq 0 (
    echo ❌ Failed to install pdfminer.six
    exit /b 1
)

echo ✅ All packages installed successfully!

echo 🧪 Testing imports...
python -c "
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
    exit /b 1
)

echo 🎉 Environment setup complete!
echo 📝 To use this environment:
echo    conda activate policy_api
echo    cd f:\Projects\MM\mm-hr-policy-forecast
echo    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

echo 💡 Environment is ready for chatbot deployment!
pause