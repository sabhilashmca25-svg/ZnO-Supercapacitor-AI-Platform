@echo off
setlocal EnableDelayedExpansion

title ZnO Supercapacitor AI Platform
cd /d "%~dp0"

echo.
powershell -NoProfile -Command "Write-Host '  ZnO Supercapacitor AI Platform  --  Setup and Launch' -ForegroundColor Cyan; Write-Host '  ============================================================' -ForegroundColor DarkCyan"
echo.

REM Venv lives on C: drive to avoid Windows App Control blocking DLLs on D:\
set "ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv"
set "FIRST_INSTALL=0"

REM ---- [1/5] Python -----------------------------------------------------------
powershell -NoProfile -Command "Write-Host '  [1/5] Checking Python ...' -ForegroundColor Cyan"
python --version >nul 2>&1
if !errorlevel! neq 0 (
    powershell -NoProfile -Command "Write-Host '  [ERROR] Python not found.' -ForegroundColor Red"
    echo.
    echo   Fix:
    echo     1. Go to https://python.org/downloads
    echo     2. Download Python 3.11 or newer
    echo     3. During install CHECK "Add Python to PATH"
    echo     4. Restart your PC, then re-run START_APP.bat
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
powershell -NoProfile -Command "Write-Host ('  [OK] ' + $env:PY_VER) -ForegroundColor Green"

REM ---- [2/5] Virtual environment ----------------------------------------------
echo.
powershell -NoProfile -Command "Write-Host '  [2/5] Checking virtual environment ...' -ForegroundColor Cyan"

if not exist "!ZNO_VENV!\Scripts\python.exe" (
    powershell -NoProfile -Command "Write-Host '        Creating virtual environment ...' -ForegroundColor Yellow"
    python -m venv "!ZNO_VENV!"
    if !errorlevel! neq 0 (
        powershell -NoProfile -Command "Write-Host '  [ERROR] Could not create virtual environment.' -ForegroundColor Red"
        echo.
        echo   Fix: Make sure Python 3.11 or newer is installed.
        echo.
        pause
        exit /b 1
    )
    set "FIRST_INSTALL=1"
)

REM Check if backend packages are installed
"!ZNO_VENV!\Scripts\python.exe" -c "import uvicorn" >nul 2>&1
if !errorlevel! equ 0 goto :venv_ready

powershell -NoProfile -Command "Write-Host '        Installing Python packages (TensorFlow ~350 MB, 10-30 min) ...' -ForegroundColor Yellow; Write-Host '        Please wait. Do NOT close this window.' -ForegroundColor DarkGray"
echo.
"!ZNO_VENV!\Scripts\python.exe" -m pip install --upgrade pip --quiet
"!ZNO_VENV!\Scripts\python.exe" -m pip install -r backend\requirements.txt
if !errorlevel! neq 0 (
    powershell -NoProfile -Command "Write-Host '  [ERROR] pip install failed.' -ForegroundColor Red"
    echo.
    echo   Command: pip install -r backend\requirements.txt
    echo   Fix: Check internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo.
echo   Unblocking extension files...
powershell -NoProfile -Command "Get-ChildItem '!ZNO_VENV!' -Recurse -Include '*.pyd','*.dll' | Unblock-File" 2>nul
powershell -NoProfile -Command "Write-Host '  [OK] Python packages installed.' -ForegroundColor Green"
set "FIRST_INSTALL=1"
goto :node_check

:venv_ready
powershell -NoProfile -Command "Write-Host '  [OK] Python packages already installed.' -ForegroundColor Green"

REM ---- [3/5] Node.js ----------------------------------------------------------
:node_check
echo.
powershell -NoProfile -Command "Write-Host '  [3/5] Checking Node.js ...' -ForegroundColor Cyan"

REM File-existence check only - no execution, no errorlevel dependency
set "NODE_DIR="
if exist "C:\Program Files\nodejs\node.exe"        set "NODE_DIR=C:\Program Files\nodejs"
if not defined NODE_DIR if exist "%LOCALAPPDATA%\Programs\nodejs\node.exe" set "NODE_DIR=%LOCALAPPDATA%\Programs\nodejs"
if not defined NODE_DIR if exist "%APPDATA%\nvm\current\node.exe"          set "NODE_DIR=%APPDATA%\nvm\current"

if not defined NODE_DIR (
    powershell -NoProfile -Command "Write-Host '  [ERROR] Node.js not found.' -ForegroundColor Red"
    echo.
    echo   Fix:
    echo     1. Go to https://nodejs.org  (LTS version)
    echo     2. Install with default options
    echo     3. Restart your PC, then re-run START_APP.bat
    echo.
    pause
    exit /b 1
)

set "PATH=!NODE_DIR!;!PATH!"
powershell -NoProfile -Command "Write-Host '  [OK] Node.js found.' -ForegroundColor Green"

REM ---- [4/5] Frontend packages ------------------------------------------------
echo.
powershell -NoProfile -Command "Write-Host '  [4/5] Checking frontend packages ...' -ForegroundColor Cyan"
if exist "frontend\node_modules\vite\bin\vite.js" goto :npm_ready

powershell -NoProfile -Command "Write-Host '        Running npm install (2-10 min first time) ...' -ForegroundColor Yellow; Write-Host '        Please wait. Do NOT close this window.' -ForegroundColor DarkGray"
echo.
pushd frontend
call npm install
if !errorlevel! neq 0 (
    popd
    powershell -NoProfile -Command "Write-Host '  [ERROR] npm install failed.' -ForegroundColor Red"
    echo.
    echo   Command: npm install  (in frontend/)
    echo   Fix: Check internet connection and try again.
    echo.
    pause
    exit /b 1
)
popd
powershell -NoProfile -Command "Write-Host '  [OK] Frontend packages installed.' -ForegroundColor Green"
set "FIRST_INSTALL=1"
goto :launch

:npm_ready
powershell -NoProfile -Command "Write-Host '  [OK] Frontend packages already installed.' -ForegroundColor Green"

REM ---- First-time install complete --------------------------------------------
:launch
if !FIRST_INSTALL! equ 1 (
    echo.
    powershell -NoProfile -Command "Write-Host '  ============================================================' -ForegroundColor Green; Write-Host '  First-time installation complete.' -ForegroundColor Green; Write-Host '  Launching application ...' -ForegroundColor Green; Write-Host '  ============================================================' -ForegroundColor Green"
)
echo.

REM ---- [5/5] Launch -----------------------------------------------------------
powershell -NoProfile -Command "Write-Host '  [5/5] Starting application ...' -ForegroundColor Cyan; Write-Host '        Browser opens automatically.' -ForegroundColor DarkGray; Write-Host '        Backend loads ML models in ~30-60 seconds.' -ForegroundColor DarkGray; Write-Host '        To stop: double-click STOP_APP.bat' -ForegroundColor DarkGray"
echo.

"!ZNO_VENV!\Scripts\python.exe" launcher.py
if !errorlevel! neq 0 (
    powershell -NoProfile -Command "Write-Host '  [ERROR] Launcher failed. See details above.' -ForegroundColor Red"
    echo.
    pause
    exit /b 1
)

echo.
powershell -NoProfile -Command "Write-Host '  Application is running. Browser window should be open.' -ForegroundColor Green; Write-Host '  Close this window any time - the app keeps running.' -ForegroundColor DarkGray"
echo.
pause
