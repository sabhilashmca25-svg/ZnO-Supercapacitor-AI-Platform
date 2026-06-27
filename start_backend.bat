@echo off
cd /d "%~dp0backend"
if not exist "venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found.
    echo  Run START_APP.bat first to set up dependencies.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause