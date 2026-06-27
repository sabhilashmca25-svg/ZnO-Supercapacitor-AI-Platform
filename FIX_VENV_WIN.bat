@echo off
title ZnO Platform - Fix Python Venv (Windows Application Control)
cd /d "%~dp0"

echo.
echo  ================================================================
echo   ZnO Supercapacitor AI Platform - Fix Python Environment
echo.
echo   Windows Application Control blocks DLLs on the D:\ drive.
echo   This script moves the Python venv to C:\LocalAppData where
echo   DLLs are trusted by Windows security policy.
echo.
echo   New venv: %LOCALAPPDATA%\ZnO_Platform_venv
echo  ================================================================
echo.

set ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv

REM -- Check Python --------------------------------------------------
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python not found in PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found: %%v

echo.

REM -- Create venv on C: drive if it doesn't exist ------------------
if exist "%ZNO_VENV%\Scripts\python.exe" (
    echo  Venv already exists at:
    echo    %ZNO_VENV%
    echo.
    "%ZNO_VENV%\Scripts\python.exe" -c "import uvicorn" >nul 2>&1
    if %errorlevel% equ 0 (
        echo  Packages already installed. Nothing to do.
        echo.
        echo  You can now use START_APP.bat normally.
        echo  ================================================================
        pause
        exit /b 0
    )
    echo  Packages missing - running pip install ...
    goto :install
)

echo  Creating Python virtual environment at:
echo    %ZNO_VENV%
echo.
python -m venv "%ZNO_VENV%"
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Failed to create virtual environment.
    echo  Make sure Python 3.11+ is installed.
    pause
    exit /b 1
)
echo  Virtual environment created.
echo.

:install
echo  Installing packages (TensorFlow ~350MB, may take 10-30 min) ...
echo  Do NOT close this window.
echo.
"%ZNO_VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
"%ZNO_VENV%\Scripts\python.exe" -m pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: pip install failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo  ================================================================
echo   Done!
echo.
echo   Restart the application with START_APP.bat
echo  ================================================================
echo.
pause
