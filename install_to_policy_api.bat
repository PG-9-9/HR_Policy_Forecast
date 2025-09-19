@echo off
REM Install packages to policy_api environment using exact working versions
echo 🚀 Installing packages to policy_api environment...

REM Check if environment exists
F:\Conda\envs\policy_api\python.exe --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ policy_api environment not found!
    echo Please create it first with: conda create -n policy_api python=3.10 -y
    pause
    exit /b 1
)

echo 📦 Installing packages with exact working versions...

REM Install all packages at once using requirements file
F:\Conda\envs\policy_api\python.exe -m pip install -r requirements-policy-api.txt

if %errorlevel% neq 0 (
    echo ❌ Package installation failed!
    pause
    exit /b 1
)

echo ✅ All packages installed successfully!

echo 🧪 Testing imports...
F:\Conda\envs\policy_api\python.exe -c "
try:
    from app.main import app
    print('✅ App imports successfully!')
except Exception as e:
    print(f'❌ App import failed: {e}')
    exit(1)
"

if %errorlevel% neq 0 (
    echo ❌ App test failed!
    pause
    exit /b 1
)

echo 🎉 policy_api environment is ready!
echo.
echo To use:
echo   conda activate policy_api
echo   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 📡 Then open browser to: http://localhost:8000
echo 💡 Note: Use localhost:8000 in browser (NOT 0.0.0.0:8000)
echo.
pause