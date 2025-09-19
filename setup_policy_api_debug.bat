@echo off
REM Debug version of environment setup for policy_api (Windows)
REM This script creates a clean environment with only chatbot dependencies

echo 🚀 Setting up policy_api environment for chatbot deployment...
echo 🔍 Debug mode: showing all command output
echo.

REM Check if conda is available
echo 📋 Checking conda installation...
conda --version
if %errorlevel% neq 0 (
    echo ❌ Conda not found! Please install Anaconda or Miniconda first
    echo Current PATH: %PATH%
    pause
    exit /b 1
)
echo ✅ Conda is available
echo.

REM Remove existing environment if it exists (ignore error if not found)
echo 📦 Checking for existing policy_api environment...
conda env list | findstr policy_api
if %errorlevel% equ 0 (
    echo Found existing policy_api environment, removing it...
    conda env remove -n policy_api -y
    if %errorlevel% neq 0 (
        echo ❌ Failed to remove existing environment
        pause
        exit /b 1
    )
) else (
    echo 💡 No existing policy_api environment found (this is normal)
)
echo.

REM Create new environment with Python 3.10
echo 🐍 Creating new conda environment: policy_api
echo Running: conda create -n policy_api python=3.10 pip -y
conda create -n policy_api python=3.10 pip -y
if %errorlevel% neq 0 (
    echo ❌ Failed to create conda environment
    echo Error level: %errorlevel%
    pause
    exit /b 1
)
echo ✅ Environment created successfully!
echo.

echo 📦 Installing packages using conda run...
echo This avoids activation issues in batch files
echo.

REM Install packages using conda run to avoid activation issues
echo [1/14] Installing fastapi...
conda run -n policy_api pip install fastapi==0.116.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install fastapi
    pause
    exit /b 1
)
echo ✅ fastapi installed

echo [2/14] Installing uvicorn...
conda run -n policy_api pip install "uvicorn[standard]==0.35.0"
if %errorlevel% neq 0 (
    echo ❌ Failed to install uvicorn
    pause
    exit /b 1
)
echo ✅ uvicorn installed

echo [3/14] Installing python-multipart...
conda run -n policy_api pip install python-multipart==0.0.20
if %errorlevel% neq 0 (
    echo ❌ Failed to install python-multipart
    pause
    exit /b 1
)
echo ✅ python-multipart installed

echo [4/14] Installing openai...
conda run -n policy_api pip install openai==1.108.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install openai
    pause
    exit /b 1
)
echo ✅ openai installed

echo [5/14] Installing jinja2...
conda run -n policy_api pip install jinja2==3.1.6
if %errorlevel% neq 0 (
    echo ❌ Failed to install jinja2
    pause
    exit /b 1
)
echo ✅ jinja2 installed

echo [6/14] Installing python-dotenv...
conda run -n policy_api pip install python-dotenv==1.1.1
if %errorlevel% neq 0 (
    echo ❌ Failed to install python-dotenv
    pause
    exit /b 1
)
echo ✅ python-dotenv installed

echo [7/14] Installing pandas...
conda run -n policy_api pip install pandas==2.3.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install pandas
    pause
    exit /b 1
)
echo ✅ pandas installed

echo [8/14] Installing numpy...
conda run -n policy_api pip install numpy==1.26.4
if %errorlevel% neq 0 (
    echo ❌ Failed to install numpy
    pause
    exit /b 1
)
echo ✅ numpy installed

echo [9/14] Installing requests...
conda run -n policy_api pip install requests==2.32.5
if %errorlevel% neq 0 (
    echo ❌ Failed to install requests
    pause
    exit /b 1
)
echo ✅ requests installed

echo [10/14] Installing sentence-transformers (this may take a while)...
conda run -n policy_api pip install sentence-transformers==5.1.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install sentence-transformers
    pause
    exit /b 1
)
echo ✅ sentence-transformers installed

echo [11/14] Installing faiss-cpu...
conda run -n policy_api pip install faiss-cpu==1.12.0
if %errorlevel% neq 0 (
    echo ❌ Failed to install faiss-cpu
    pause
    exit /b 1
)
echo ✅ faiss-cpu installed

echo [12/14] Installing rank-bm25...
conda run -n policy_api pip install rank-bm25==0.2.2
if %errorlevel% neq 0 (
    echo ❌ Failed to install rank-bm25
    pause
    exit /b 1
)
echo ✅ rank-bm25 installed

echo [13/14] Installing beautifulsoup4...
conda run -n policy_api pip install beautifulsoup4==4.13.5
if %errorlevel% neq 0 (
    echo ❌ Failed to install beautifulsoup4
    pause
    exit /b 1
)
echo ✅ beautifulsoup4 installed

echo [14/14] Installing pdfminer.six...
conda run -n policy_api pip install pdfminer.six==20250506
if %errorlevel% neq 0 (
    echo ❌ Failed to install pdfminer.six
    pause
    exit /b 1
)
echo ✅ pdfminer.six installed

echo.
echo ✅ All packages installed successfully!
echo.

echo 🧪 Testing imports...
conda run -n policy_api python -c "
try:
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
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ Import test failed!
    pause
    exit /b 1
)

echo.
echo 🎉 Environment setup complete!
echo.
echo 📝 To use this environment:
echo    conda activate policy_api
echo    cd f:\Projects\MM\mm-hr-policy-forecast
echo    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 💡 Environment is ready for chatbot deployment!
echo.
echo Press any key to continue...
pause