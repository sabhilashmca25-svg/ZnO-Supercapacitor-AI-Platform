#!/usr/bin/env bash
# ZnO Supercapacitor AI Platform — Setup & Launch (macOS / Linux)
# Usage: bash start_app.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo " ================================================================"
echo "  ZnO Supercapacitor AI Platform  |  Setup & Launch"
echo " ================================================================"
echo ""

# -- [1/5] Check Python ---------------------------------------------------
echo " [1/5] Checking Python ..."
if ! command -v python3 &>/dev/null; then
    echo ""
    echo "  ERROR: Python 3 not found."
    echo "  1. Go to https://python.org/downloads"
    echo "  2. Download Python 3.11 or newer"
    echo "  3. Re-run:  bash start_app.sh"
    echo ""
    exit 1
fi
echo "  Found: $(python3 --version)"

# -- [2/5] Python virtual environment ------------------------------------
echo ""
echo " [2/5] Checking Python environment ..."
VENV_DIR="$SCRIPT_DIR/backend/venv"
VENV_PY="$VENV_DIR/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "  Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
fi

if ! "$VENV_PY" -c "import uvicorn" &>/dev/null 2>&1; then
    echo "  Installing packages (TensorFlow ~350MB, may take 10-30 min) ..."
    echo "  Please wait and do NOT close this terminal."
    echo ""
    "$VENV_PY" -m pip install --upgrade pip --quiet
    "$VENV_PY" -m pip install -r "$SCRIPT_DIR/backend/requirements.txt"
    echo ""
    echo "  Python environment ready!"
else
    echo "  Already installed - skipping."
fi

# -- [3/5] Check Node.js -------------------------------------------------
echo ""
echo " [3/5] Checking Node.js ..."
if ! command -v node &>/dev/null; then
    echo ""
    echo "  ERROR: Node.js not found."
    echo "  1. Go to https://nodejs.org  (LTS version)"
    echo "  2. Install with default options"
    echo "  3. Re-run:  bash start_app.sh"
    echo ""
    exit 1
fi
echo "  Found: Node.js $(node --version)"
echo "  Found: npm v$(npm --version)"

# -- [4/5] Frontend dependencies -----------------------------------------
echo ""
echo " [4/5] Checking frontend dependencies ..."
if [ ! -f "$SCRIPT_DIR/frontend/node_modules/vite/bin/vite.js" ]; then
    echo "  Running npm install (2-10 min on first run) ..."
    echo ""
    pushd "$SCRIPT_DIR/frontend" > /dev/null
    npm install
    popd > /dev/null
    echo ""
    echo "  Frontend ready!"
else
    echo "  Already installed - skipping."
fi

# -- [5/5] Launch --------------------------------------------------------
echo ""
echo " ================================================================"
echo "  Starting application ..."
echo "  Browser opens automatically."
echo "  Backend loads ML models in ~30-60 seconds."
echo "  To stop the app: press Ctrl+C or run: python3 stop_app.py"
echo " ================================================================"
echo ""

# Trap Ctrl+C so we cleanly stop the backend and frontend
cleanup() {
    echo ""
    echo " Stopping application ..."
    "$VENV_PY" "$SCRIPT_DIR/stop_app.py" 2>/dev/null || true
    echo " Done."
    exit 0
}
trap cleanup INT TERM

# Launch the app (launcher.py starts backend + frontend, then returns)
"$VENV_PY" "$SCRIPT_DIR/launcher.py"

echo ""
echo " ================================================================"
echo "  Application is running."
echo "  Open browser at:  http://localhost:5173"
echo "  Press Ctrl+C to stop the application."
echo " ================================================================"
echo ""

# Keep this script alive so Ctrl+C is caught
while true; do
    sleep 30
    # Exit if launcher lock is gone (app was stopped externally)
    if [ ! -f "$SCRIPT_DIR/.launcher.lock" ]; then
        echo " Application has stopped."
        break
    fi
done
