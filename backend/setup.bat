@echo off
REM ─────────────────────────────────────────────────────────────────
REM  ZnO Supercapacitor AI Platform — One-click Setup
REM
REM  WHAT THIS DOES:
REM    1. Creates a Python virtual environment in backend/venv/
REM    2. Installs all required packages from requirements.txt
REM
REM  PREREQUISITES:
REM    Python 3.11, 3.12, or 3.13 must be installed.
REM    Download from https://python.org
REM
REM  USAGE:
REM    cd backend
REM    setup.bat
REM ─────────────────────────────────────────────────────────────────

echo.
echo  ============================================================
echo   ZnO Supercapacitor AI Platform — Setup
echo  ============================================================
echo.

REM Check Python exists
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python not found in PATH.
    echo  Please install Python from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo  Found: %%i

echo.

REM Create venv if it doesn't exist
if not exist "venv\Scripts\python.exe" (
    echo  Creating virtual environment...
    python -m venv venv
    echo  Virtual environment created.
) else (
    echo  Virtual environment already exists.
)

echo.

REM Activate and install
call venv\Scripts\activate.bat
echo  Installing packages from requirements.txt...
echo  (TensorFlow is ~350 MB - this may take 10-30 minutes)
echo.
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Package installation failed.
    echo  Check the error messages above.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Setup complete!
echo.
echo   To start the server, run:
echo     start_server.bat
echo.
echo   Or manually:
echo     venv\Scripts\activate.bat
echo     uvicorn app.main:app --reload --port 8000
echo  ============================================================
echo.
pause
