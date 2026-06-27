@echo off
REM ─────────────────────────────────────────────────────────────────
REM  ZnO Supercapacitor AI Platform — One-click Setup
REM
REM  WHAT THIS DOES:
REM    1. Creates a Python virtual environment in %LOCALAPPDATA%\ZnO_Platform_venv
REM       (C: drive avoids Windows Application Control blocking DLLs on D:\)
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

set ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv

REM Create venv if it doesn't exist
if not exist "%ZNO_VENV%\Scripts\python.exe" (
    echo  Creating virtual environment at:
    echo    %ZNO_VENV%
    python -m venv "%ZNO_VENV%"
    echo  Virtual environment created.
) else (
    echo  Virtual environment already exists at:
    echo    %ZNO_VENV%
)

echo.

REM Activate and install
call "%ZNO_VENV%\Scripts\activate.bat"
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
echo     call %ZNO_VENV%\Scripts\activate.bat
echo     uvicorn app.main:app --reload --port 8000
echo  ============================================================
echo.
pause
