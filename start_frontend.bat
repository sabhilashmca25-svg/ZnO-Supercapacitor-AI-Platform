@echo off
cd /d "%~dp0frontend"
if not exist "node_modules\vite\bin\vite.js" (
    echo  ERROR: Frontend dependencies not installed.
    echo  Run START_APP.bat first to set up dependencies.
    pause
    exit /b 1
)
npm run dev
pause