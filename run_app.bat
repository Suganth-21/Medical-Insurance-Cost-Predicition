@echo off
echo Starting Medical Insurance Cost Predictor...

:: Activate Virtual Environment
if not exist venv (
    echo Error: Virtual environment 'venv' not found!
    echo Please run setup first.
    pause
    exit /b 1
)

call venv\Scripts\activate

echo.
echo ==================================================
echo [1/2] Starting FastAPI Backend and Frontend Server
echo ==================================================
start "FastAPI Server" cmd /c "uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo.
echo ==================================================
echo [2/2] Opening Application in Default Browser
echo ==================================================
timeout /t 3 >nul
start http://localhost:8000/

echo.
echo Application is running at: http://localhost:8000/
echo Close this window to stop the servers if you didn't run them as background tasks.
echo.
pause
