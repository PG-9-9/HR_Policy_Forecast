@echo off
REM Complete automated setup for policy_api chatbot environment
REM Creates environment, installs packages, validates, and starts server

echo 🤖 AUTOMATED POLICY_API CHATBOT SETUP
echo =====================================

REM Step 1: Create environment
echo 📦 Step 1: Creating policy_api environment...
call setup_policy_api_env_fixed.bat
if %errorlevel% neq 0 (
    echo ❌ Environment setup failed!
    pause
    exit /b 1
)

echo.
echo ✅ Environment created successfully!
echo.

REM Step 2: Activate environment
echo ⚡ Step 2: Activating policy_api environment...
call conda activate policy_api

REM Step 3: Validate dependencies
echo 🔍 Step 3: Validating dependencies...
python check_and_install_deps.py
if %errorlevel% neq 0 (
    echo ❌ Dependency validation failed!
    pause
    exit /b 1
)

echo.
echo ✅ All dependencies validated!
echo.

REM Step 4: Test app import
echo 🧪 Step 4: Testing app import...
python -c "
try:
    from app.main import app
    print('✅ App imports successfully - ready for deployment!')
except Exception as e:
    print(f'❌ App import failed: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ App test failed!
    pause
    exit /b 1
)

echo.
echo 🎉 SETUP COMPLETE!
echo.
echo 📝 Summary:
echo   ✅ policy_api environment created
echo   ✅ All chatbot dependencies installed
echo   ✅ App imports successfully
echo   ✅ Ready for AWS deployment
echo.
echo 🚀 To start the server:
echo   conda activate policy_api
echo   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 🐳 To build Docker image:
echo   docker build -t mm-hr-chat-linux .
echo.

set /p start_server=Start the server now? (y/n): 
if /i "%start_server%"=="y" (
    echo 🚀 Starting FastAPI server...
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) else (
    echo 💡 Run 'test_and_start_app.bat' when ready to start the server
    pause
)