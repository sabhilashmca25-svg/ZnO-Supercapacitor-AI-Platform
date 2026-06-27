#!/usr/bin/env bash
# ZnO Supercapacitor AI Platform — Setup & Launch (macOS / Linux)
# Usage: bash start_app.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GRN='\033[0;92m'; CYN='\033[0;96m'; YLW='\033[0;93m'
RED='\033[0;91m'; DIM='\033[0;90m'; BLD='\033[1m'; RST='\033[0m'

echo ""
printf "${BLD}${CYN} ================================================================${RST}\n"
printf "${BLD}${CYN}   ZnO Supercapacitor AI Platform  |  Setup & Launch${RST}\n"
printf "${BLD}${CYN} ================================================================${RST}\n"
echo ""

FIRST_INSTALL=0

# ── [1/5] Python ──────────────────────────────────────────────────────────────
printf "${CYN} >> Checking Python ...${RST}\n"
if ! command -v python3 &>/dev/null; then
    printf "${RED} [ERROR] Python 3 not found.${RST}\n"
    echo ""
    echo "   Fix:"
    echo "     1. Go to https://python.org/downloads"
    echo "     2. Download Python 3.11 or newer"
    echo "     3. Re-run:  bash start_app.sh"
    echo ""
    exit 1
fi
printf "${GRN} [OK] $(python3 --version)${RST}\n"

# ── [2/5] Virtual environment ─────────────────────────────────────────────────
echo ""
printf "${CYN} >> Checking virtual environment ...${RST}\n"
VENV_DIR="$SCRIPT_DIR/backend/venv"
VENV_PY="$VENV_DIR/bin/python"

if [ ! -f "$VENV_PY" ]; then
    printf "${YLW}      Creating virtual environment ...${RST}\n"
    python3 -m venv "$VENV_DIR"
    FIRST_INSTALL=1
fi

if ! "$VENV_PY" -c "import uvicorn" &>/dev/null 2>&1; then
    printf "${YLW}      Installing Python packages (TensorFlow ~350 MB — 10-30 min) ...${RST}\n"
    printf "${DIM}      Please wait. Do NOT close this terminal.${RST}\n"
    echo ""
    "$VENV_PY" -m pip install --upgrade pip --quiet
    "$VENV_PY" -m pip install -r "$SCRIPT_DIR/backend/requirements.txt"
    printf "${GRN} [OK] Python packages installed.${RST}\n"
    FIRST_INSTALL=1
else
    printf "${GRN} [OK] Python packages already installed.${RST}\n"
fi

# ── [3/5] Node.js ─────────────────────────────────────────────────────────────
echo ""
printf "${CYN} >> Checking Node.js ...${RST}\n"
if ! command -v node &>/dev/null; then
    printf "${RED} [ERROR] Node.js not found.${RST}\n"
    echo ""
    echo "   Fix:"
    echo "     1. Go to https://nodejs.org  (LTS version)"
    echo "     2. Install with default options"
    echo "     3. Re-run:  bash start_app.sh"
    echo ""
    exit 1
fi
printf "${GRN} [OK] Node.js $(node --version)   npm v$(npm --version)${RST}\n"

# ── [4/5] Frontend packages ───────────────────────────────────────────────────
echo ""
printf "${CYN} >> Checking frontend packages ...${RST}\n"
if [ ! -f "$SCRIPT_DIR/frontend/node_modules/vite/bin/vite.js" ]; then
    printf "${YLW}      Running npm install (2-10 min first time) ...${RST}\n"
    printf "${DIM}      Please wait. Do NOT close this terminal.${RST}\n"
    echo ""
    pushd "$SCRIPT_DIR/frontend" > /dev/null
    npm install
    popd > /dev/null
    printf "${GRN} [OK] Frontend packages installed.${RST}\n"
    FIRST_INSTALL=1
else
    printf "${GRN} [OK] Frontend packages already installed.${RST}\n"
fi

# ── First-time install complete ───────────────────────────────────────────────
if [ "$FIRST_INSTALL" -eq 1 ]; then
    echo ""
    printf "${GRN} ================================================================${RST}\n"
    printf "${GRN}  First-time installation completed successfully.${RST}\n"
    printf "${GRN}  Launching application ...${RST}\n"
    printf "${GRN} ================================================================${RST}\n"
fi
echo ""

# ── [5/5] Launch ──────────────────────────────────────────────────────────────
printf "${CYN} >> Starting application ...${RST}\n"
printf "${DIM}      Browser opens automatically.${RST}\n"
printf "${DIM}      Backend loads ML models in ~30-60 seconds.${RST}\n"
printf "${DIM}      Press Ctrl+C to stop.${RST}\n"
echo ""

cleanup() {
    echo ""
    echo " Stopping application ..."
    "$VENV_PY" "$SCRIPT_DIR/stop_app.py" 2>/dev/null || true
    echo " Done."
    exit 0
}
trap cleanup INT TERM

"$VENV_PY" "$SCRIPT_DIR/launcher.py"

echo ""
printf "${GRN} Application is running. Browser window should be open.${RST}\n"
printf "${DIM} Press Ctrl+C to stop the application.${RST}\n"
echo ""

while true; do
    sleep 30
    if [ ! -f "$SCRIPT_DIR/.launcher.lock" ]; then
        echo " Application has stopped."
        break
    fi
done
