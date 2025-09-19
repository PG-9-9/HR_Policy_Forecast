@echo off
REM Simple environment creation script for policy_api
echo 🚀 Creating policy_api environment...

REM Check conda availability
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda not available
    echo Please ensure conda is installed and in your PATH
    pause
    exit /b 1
)

REM Create environment
echo 📦 Creating conda environment...
conda create -n policy_api python=3.10 -y

REM Check if creation was successful
if %errorlevel% neq 0 (
    echo ❌ Environment creation failed
    pause
    exit /b 1
)

echo ✅ Environment created successfully!
echo.
echo 📝 Next steps:
echo 1. Activate the environment:
echo    conda activate policy_api
echo.
echo 2. Install required packages:
echo    pip install fastapi uvicorn[standard] python-multipart openai
echo    pip install jinja2 python-dotenv pandas numpy requests
echo    pip install sentence-transformers faiss-cpu rank-bm25
echo    pip install beautifulsoup4 pdfminer.six
echo.
echo 3. Test the app:
echo    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 💡 Or run the automated installer:
echo    python check_and_install_deps.py
echo.
pause