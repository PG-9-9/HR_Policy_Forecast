@echo off
REM Build and compare different Docker image sizes

echo 🐳 Building different Docker image sizes...
echo.

echo 📊 Building images:
echo 1. Ultra-minimal (no ML) - ~200-400MB
echo 2. CPU-only (with ML) - ~1-2GB 
echo 3. Current (with CUDA) - ~11GB
echo.

echo ⏱️ Building ultra-minimal image...
docker build -t mm-hr-ultra-minimal -f Dockerfile.ultra-minimal . 2>nul
if %errorlevel% equ 0 (
    echo ✅ Ultra-minimal build complete
) else (
    echo ❌ Ultra-minimal build failed
)

echo.
echo ⏱️ Building CPU-only image...
docker build -t mm-hr-cpu-only -f Dockerfile.minimal . 2>nul
if %errorlevel% equ 0 (
    echo ✅ CPU-only build complete
) else (
    echo ❌ CPU-only build failed
)

echo.
echo ⏱️ Building current image...
docker build -t mm-hr-current . 2>nul
if %errorlevel% equ 0 (
    echo ✅ Current build complete
) else (
    echo ❌ Current build failed
)

echo.
echo 📏 Image sizes:
docker images | findstr mm-hr

echo.
echo 🧪 Testing images:
echo.

echo Testing ultra-minimal...
docker run --rm -d -p 8001:8000 --name test-ultra mm-hr-ultra-minimal 2>nul
timeout /t 5 >nul
curl -s http://localhost:8001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ultra-minimal: Working
) else (
    echo ❌ Ultra-minimal: Failed
)
docker stop test-ultra >nul 2>&1

echo Testing CPU-only...
docker run --rm -d -p 8002:8000 --name test-cpu mm-hr-cpu-only 2>nul
timeout /t 10 >nul
curl -s http://localhost:8002/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ CPU-only: Working
) else (
    echo ❌ CPU-only: Failed
)
docker stop test-cpu >nul 2>&1

echo Testing current...
docker run --rm -d -p 8003:8000 --name test-current mm-hr-current 2>nul
timeout /t 15 >nul
curl -s http://localhost:8003/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Current: Working
) else (
    echo ❌ Current: Failed
)
docker stop test-current >nul 2>&1

echo.
echo 💡 Recommendation: Use ultra-minimal for OpenAI-only chat, CPU-only for local RAG
pause