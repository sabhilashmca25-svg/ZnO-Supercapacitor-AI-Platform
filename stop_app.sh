#!/usr/bin/env bash
# ZnO Supercapacitor AI Platform — Stop App (macOS / Linux)
# Usage: bash stop_app.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/backend/venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "ERROR: Virtual environment not found. Run start_app.sh first."
    exit 1
fi

"$VENV_PY" "$SCRIPT_DIR/stop_app.py"
