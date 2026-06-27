@echo off
title ZnO Supercapacitor AI Platform - Stop
color 0C
cd /d "%~dp0"

echo.
echo  ================================================================
echo   ZnO Supercapacitor AI Platform  ^|  Stop
echo  ================================================================
echo.

set ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv
if exist "%ZNO_VENV%\Scripts\python.exe" (
    "%ZNO_VENV%\Scripts\python.exe" stop_app.py
) else if exist "backend\venv\Scripts\python.exe" (
    backend\venv\Scripts\python.exe stop_app.py
) else (
    python stop_app.py 2>nul
    if %errorlevel% neq 0 (
        echo   Python not found and no venv exists.
        echo   Close the ZnO Backend and ZnO Frontend windows manually.
        echo.
        pause
    )
)