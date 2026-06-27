@echo off
cd /d "%~dp0backend"
set ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv
if not exist "%ZNO_VENV%\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found at %ZNO_VENV%
    echo  Run FIX_VENV_WIN.bat (or START_APP.bat) from the project root first.
    pause
    exit /b 1
)
call "%ZNO_VENV%\Scripts\activate.bat"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
