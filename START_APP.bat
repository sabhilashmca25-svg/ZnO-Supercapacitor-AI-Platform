@echo off
REM ─────────────────────────────────────────────────────────────────
REM  ZnO Supercapacitor AI Platform — Start Everything
REM  Double-click this file to launch both backend + frontend
REM
REM  Then open in browser:
REM    PC      →  http://localhost:5173
REM    Phone   →  http://192.168.1.3:5173  (same WiFi required)
REM ─────────────────────────────────────────────────────────────────

echo.
echo  ============================================================
echo   ZnO Supercapacitor AI Platform
echo  ============================================================
echo.
echo   PC     : http://localhost:5173
echo   Phone  : http://192.168.1.3:5173
echo.
echo   Starting backend ...
echo  ============================================================
echo.

REM ── Start Backend in its own window ──────────────────────────────
start "ZnO Backend (port 8000)" cmd /k "cd /d "%~dp0backend" && call venv\Scripts\activate.bat && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM ── Give backend 3 seconds to boot before frontend starts ────────
timeout /t 3 /nobreak > nul

REM ── Start Frontend in its own window ─────────────────────────────
start "ZnO Frontend (port 5173)" cmd /k "cd /d "%~dp0frontend" && npm run dev -- --host 0.0.0.0"

echo.
echo  Both servers are starting in separate windows.
echo  Wait ~10 seconds then open your browser.
echo.
echo   PC     : http://localhost:5173
echo   Phone  : http://192.168.1.3:5173
echo.
pause
