@echo off
REM ─────────────────────────────────────────────────────────────────
REM  ZnO Supercapacitor AI Platform — Start Backend Server
REM
REM  USAGE: Double-click this file OR run from terminal:
REM     cd backend
REM     start_server.bat
REM
REM  Requirements: Run "pip install -r requirements.txt" first
REM  (or run setup.bat to do it automatically)
REM ─────────────────────────────────────────────────────────────────

echo.
echo  ============================================================
echo   ZnO Supercapacitor AI Platform — Backend Server
echo  ============================================================
echo.

REM Check that venv exists
if not exist "venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment not found.
    echo  Please run setup.bat first to create the venv.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo  Virtual environment activated.
echo  Starting FastAPI server on http://localhost:8000
echo.
echo  Swagger UI : http://localhost:8000/docs
echo  Health API : http://localhost:8000/api/v1/health
echo.
echo  Press CTRL+C to stop the server.
echo  ============================================================
echo.

REM Start uvicorn
python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

pause
