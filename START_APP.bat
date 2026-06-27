@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM ── Enable ANSI/VT100 colours (Windows 10 1607+) ──────────────────────────────
for /f %%a in ('echo prompt $E^| cmd /k exit') do set "ESC=%%a"
set "GRN=!ESC![92m"
set "CYN=!ESC![96m"
set "YLW=!ESC![93m"
set "RED=!ESC![91m"
set "DIM=!ESC![90m"
set "BLD=!ESC![1m"
set "RST=!ESC![0m"

title ZnO Supercapacitor AI Platform
cd /d "%~dp0"

echo.
echo !BLD!!CYN! ================================================================!RST!
echo !BLD!!CYN!   ZnO Supercapacitor AI Platform  ^|  Setup ^& Launch!RST!
echo !BLD!!CYN! ================================================================!RST!
echo.

REM Venv on C: drive avoids Windows Application Control blocking DLLs on D:\
set "ZNO_VENV=%LOCALAPPDATA%\ZnO_Platform_venv"
set "FIRST_INSTALL=0"

REM ── [1/5] Python ─────────────────────────────────────────────────────────────
echo !CYN!  >> Checking Python ...!RST!
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo !RED!  [ERROR] Python is not installed or not in PATH.!RST!
    echo.
    echo         Fix:
    echo           1. Go to https://python.org/downloads
    echo           2. Download Python 3.11 or newer
    echo           3. During install, CHECK "Add Python to PATH"
    echo           4. Restart your computer
    echo           5. Re-run START_APP.bat
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo !GRN!  [OK] !PY_VER!!RST!

REM ── [2/5] Virtual environment ─────────────────────────────────────────────────
echo.
echo !CYN!  >> Checking virtual environment ...!RST!

if not exist "%ZNO_VENV%\Scripts\python.exe" (
    echo !YLW!       Creating virtual environment at!RST!
    echo !DIM!       %ZNO_VENV%!RST!
    python -m venv "%ZNO_VENV%"
    if !errorlevel! neq 0 (
        echo !RED!  [ERROR] Could not create virtual environment.!RST!
        echo.
        echo         Fix: Make sure Python 3.11 or newer is installed.
        echo         Command: python -m venv "%ZNO_VENV%"
        echo.
        pause
        exit /b 1
    )
    set "FIRST_INSTALL=1"
)

REM Check if packages are installed
"%ZNO_VENV%\Scripts\python.exe" -c "import uvicorn" >nul 2>&1
if !errorlevel! equ 0 goto :venv_ready

echo !YLW!       Installing Python packages!RST!
echo !DIM!       (TensorFlow ~350 MB — 10-30 min on first run)!RST!
echo !DIM!       Please wait. Do NOT close this window.!RST!
echo.
"%ZNO_VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet
"%ZNO_VENV%\Scripts\python.exe" -m pip install -r backend\requirements.txt
if !errorlevel! neq 0 (
    echo.
    echo !RED!  [ERROR] pip install failed.!RST!
    echo.
    echo         Command: pip install -r backend\requirements.txt
    echo         Fix: Check internet connection and try again.
    echo.
    pause
    exit /b 1
)
REM Unblock DLLs (removes Zone.Identifier on downloaded files)
echo.
echo !DIM!       Unblocking extension files ...!RST!
powershell -NoProfile -Command "Get-ChildItem '%ZNO_VENV%' -Recurse -Include '*.pyd','*.dll' | Unblock-File" 2>nul
echo !GRN!  [OK] Python packages installed.!RST!
set "FIRST_INSTALL=1"
goto :node_check

:venv_ready
echo !GRN!  [OK] Python packages already installed.!RST!

REM ── [3/5] Node.js ─────────────────────────────────────────────────────────────
:node_check
echo.
echo !CYN!  >> Checking Node.js ...!RST!
node --version >nul 2>&1
if !errorlevel! neq 0 (
    echo !RED!  [ERROR] Node.js not found.!RST!
    echo.
    echo         Fix:
    echo           1. Go to https://nodejs.org  (choose LTS version)
    echo           2. Install with default options
    echo           3. Re-run START_APP.bat
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do set "NODE_VER=%%v"
for /f "tokens=*" %%v in ('npm --version 2^>^&1')  do set "NPM_VER=%%v"
echo !GRN!  [OK] Node.js !NODE_VER!   npm v!NPM_VER!!RST!

REM ── [4/5] Frontend packages ───────────────────────────────────────────────────
echo.
echo !CYN!  >> Checking frontend packages ...!RST!
if exist "frontend\node_modules\vite\bin\vite.js" goto :npm_ready

echo !YLW!       Running npm install!RST!
echo !DIM!       (2-10 min on first run)!RST!
echo !DIM!       Please wait. Do NOT close this window.!RST!
echo.
pushd frontend
call npm install
if !errorlevel! neq 0 (
    popd
    echo.
    echo !RED!  [ERROR] npm install failed.!RST!
    echo.
    echo         Command: npm install  (in frontend/)
    echo         Fix: Check internet connection and try again.
    echo.
    pause
    exit /b 1
)
popd
echo !GRN!  [OK] Frontend packages installed.!RST!
set "FIRST_INSTALL=1"
goto :launch

:npm_ready
echo !GRN!  [OK] Frontend packages already installed.!RST!

REM ── First-time install complete ───────────────────────────────────────────────
:launch
if !FIRST_INSTALL! equ 1 (
    echo.
    echo !GRN! ================================================================!RST!
    echo !GRN!  First-time installation completed successfully.!RST!
    echo !GRN!  Launching application ...!RST!
    echo !GRN! ================================================================!RST!
)
echo.

REM ── [5/5] Launch ──────────────────────────────────────────────────────────────
echo !CYN!  >> Starting application ...!RST!
echo !DIM!       Browser opens automatically.!RST!
echo !DIM!       Backend loads ML models in ~30-60 seconds.!RST!
echo !DIM!       To stop: double-click STOP_APP.bat!RST!
echo.

"%ZNO_VENV%\Scripts\python.exe" launcher.py
if !errorlevel! neq 0 (
    echo.
    echo !RED!  [ERROR] Launcher failed.!RST!
    echo.
    echo         Command: "%ZNO_VENV%\Scripts\python.exe" launcher.py
    echo         See error details above.
    echo.
    pause
    exit /b 1
)

echo.
echo !GRN!  Application is running. Browser window should be open.!RST!
echo !DIM!  Close this window any time — the app keeps running.!RST!
echo.
pause
