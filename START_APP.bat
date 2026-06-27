@echo off
title ZnO Supercapacitor AI Platform - Setup and Start
cd /d "%~dp0"

echo.
echo  ================================================================
echo   ZnO Supercapacitor AI Platform  ^|  Setup ^& Launch
echo  ================================================================
echo.

REM Venv lives on C: drive to avoid Windows Application Control blocking
REM DLLs on non-system drives (scipy/sklearn .pyd files on D:\ are blocked).
set ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv

REM -- [1/5] Check Python -------------------------------------------
echo  [1/5] Checking Python ...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python is not installed or not found in PATH.
    echo.
    echo  1. Go to https://python.org/downloads
    echo  2. Download Python 3.11 or newer
    echo  3. During install, CHECK "Add Python to PATH"
    echo  4. Restart your computer, then re-run START_APP.bat
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found: %%v

REM -- [2/5] Python virtual environment ----------------------------
echo.
echo  [2/5] Checking Python environment ...

if not exist "%ZNO_VENV%\Scripts\python.exe" (
    echo   Creating virtual environment at %ZNO_VENV% ...
    python -m venv "%ZNO_VENV%"
    if %errorlevel% neq 0 (
        echo.
        echo  ERROR: Could not create virtual environment.
        echo  Make sure Python 3.11 or newer is installed.
        pause
        exit /b 1
    )
)

REM Use goto to avoid nested-block errorlevel parse-time expansion bug
"%ZNO_VENV%\Scripts\python.exe" -c "import uvicorn" >nul 2>&1
if %errorlevel% equ 0 goto :venv_ready

echo   Installing packages ^(TensorFlow ~350MB, may take 10-30 min^) ...
echo   Please wait and do NOT close this window.
echo.
"%ZNO_VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
"%ZNO_VENV%\Scripts\python.exe" -m pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: pip install failed.
    echo  Check internet connection and try again.
    pause
    exit /b 1
)

REM Unblock DLLs from Zone.Identifier restriction (some Windows security policies)
echo.
echo   Unblocking Python extension files ...
powershell -NoProfile -Command "Get-ChildItem '%ZNO_VENV%' -Recurse -Include '*.pyd','*.dll' | Unblock-File" 2>nul
echo.
echo   Python environment ready!
goto :node_check

:venv_ready
echo   Already installed - skipping.

REM -- [3/5] Check Node.js -----------------------------------------
:node_check
echo.
echo  [3/5] Checking Node.js ...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Node.js not found.
    echo.
    echo  1. Go to https://nodejs.org  ^(LTS version^)
    echo  2. Install with default options
    echo  3. Re-run START_APP.bat
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do echo   Found: Node.js %%v
for /f "tokens=*" %%v in ('npm --version 2^>^&1')  do echo   Found: npm v%%v

REM -- [4/5] Frontend dependencies ---------------------------------
echo.
echo  [4/5] Checking frontend dependencies ...
if exist "frontend\node_modules\vite\bin\vite.js" goto :npm_ready

echo   Running npm install ^(2-10 min on first run^) ...
echo.
pushd frontend
npm install
if %errorlevel% neq 0 (
    popd
    echo.
    echo  ERROR: npm install failed.
    echo  Check internet connection and try again.
    pause
    exit /b 1
)
popd
echo.
echo   Frontend ready!
goto :launch

:npm_ready
echo   Already installed - skipping.

REM -- [5/5] Launch ------------------------------------------------
:launch
echo.
echo  ================================================================
echo   Starting application ...
echo   Browser opens automatically.
echo   Backend loads ML models in ~30-60 seconds.
echo   Local URL : http://localhost:5173
echo   Stop app  : double-click STOP_APP.bat
echo  ================================================================
echo.

"%ZNO_VENV%\Scripts\python.exe" launcher.py
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Launcher failed. See details above.
    pause
    exit /b 1
)
pause
