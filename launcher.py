"""
ZnO Supercapacitor AI Platform — Smart Launcher

Finds free ports dynamically so this project can run alongside other
Vite/FastAPI projects without port conflicts.  Safe to double-click
any number of times — a running instance is detected and its browser
window is brought up instead of spawning duplicates.

Cross-platform: works on Windows, macOS, and Linux.
Lock file:  .launcher.lock   (auto-created, auto-removed by stop_app.py)
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"

# Enable VT100/ANSI colours on Windows 10+ CMD (no-op on macOS/Linux)
if IS_WINDOWS:
    os.system("")

GRN = "\033[92m"
CYN = "\033[96m"
YLW = "\033[93m"
RED = "\033[91m"
DIM = "\033[90m"
BLD = "\033[1m"
RST = "\033[0m"

# Chrome paths — Windows and macOS
if IS_WINDOWS:
    CHROME_PATHS = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
else:
    CHROME_PATHS = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]


def open_app_window(url: str) -> None:
    """Open URL in Chrome standalone app window; fall back to default browser."""
    for chrome in CHROME_PATHS:
        if Path(chrome).exists():
            kwargs: dict = {}
            if IS_WINDOWS:
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen([chrome, f"--app={url}"], **kwargs)
            return
    webbrowser.open(url)


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
BACKEND_DIR  = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
VITE_JS      = FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"

# Windows: prefer venv on C: drive (%LOCALAPPDATA%) to avoid Windows Application
# Control blocking scipy/sklearn DLLs on non-system drives (D:\, E:\, etc.).
# macOS/Linux: always use backend/venv/bin/python.
if IS_WINDOWS:
    _C_VENV = Path(os.environ.get("LOCALAPPDATA", "")) / "ZnO_Platform_venv"
    VENV_PYTHON = (
        _C_VENV / "Scripts" / "python.exe"
        if (_C_VENV / "Scripts" / "python.exe").exists()
        else BACKEND_DIR / "venv" / "Scripts" / "python.exe"
    )
else:
    VENV_PYTHON = BACKEND_DIR / "venv" / "bin" / "python"

LOCK_FILE = ROOT / ".launcher.lock"

# ── Port search start ─────────────────────────────────────────────────────────
BACKEND_PORT_START  = 8000
FRONTEND_PORT_START = 5173

# ── Timeouts (seconds) ────────────────────────────────────────────────────────
FRONTEND_TIMEOUT = 30
BACKEND_TIMEOUT  = 120   # RF model takes ~45 s to load

SEP = "-" * 54


# ──────────────────────────────────────────────────────────────────────────────
# Network helpers
# ──────────────────────────────────────────────────────────────────────────────

def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_free_port(start: int) -> int:
    p = start
    while port_in_use(p):
        p += 1
    return p


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_url(url: str, timeout: int, interval: float = 2.0) -> bool:
    """Poll url until HTTP 200 or timeout expires.  Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_ok(url):
            return True
        time.sleep(interval)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Lock file
# ──────────────────────────────────────────────────────────────────────────────

def write_lock(backend_port: int, frontend_port: int,
               backend_pid: int, frontend_pid: int) -> None:
    LOCK_FILE.write_text(json.dumps({
        "backend_port":  backend_port,
        "frontend_port": frontend_port,
        "backend_pid":   backend_pid,
        "frontend_pid":  frontend_pid,
    }, indent=2), encoding="utf-8")


def read_lock() -> dict | None:
    try:
        return json.loads(LOCK_FILE.read_text(encoding="utf-8")) if LOCK_FILE.exists() else None
    except Exception:
        return None


def clear_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Process helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_pid_alive(pid: int) -> bool:
    """True if the process with this PID is still running (cross-platform)."""
    if IS_WINDOWS:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in r.stdout
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def check_existing() -> dict | None:
    """
    Return lock data if this project is already running.
    Removes a stale lock file when the processes have died.

    Two-stage check:
      1. Both PIDs must be alive (fast, OS-level check).
      2. Backend port must actually respond to HTTP (guards against PID reuse
         where an unrelated process happens to have the same PID as a previous
         backend, causing a false-positive "already running" detection).
    """
    data = read_lock()
    if data is None:
        return None

    b_pid  = data.get("backend_pid")
    f_pid  = data.get("frontend_pid")
    b_port = data.get("backend_port", BACKEND_PORT_START)

    if b_pid and f_pid and is_pid_alive(b_pid) and is_pid_alive(f_pid):
        # Verify the backend port is actually ours, not a reused PID
        if http_ok(f"http://localhost:{b_port}/api/v1/health"):
            return data
        # PIDs alive but backend not responding — stale lock
        clear_lock()
        return None

    clear_lock()
    return None


def _popen_kwargs() -> dict:
    """Extra Popen kwargs to hide console windows on Windows; empty on macOS/Linux."""
    if not IS_WINDOWS:
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return {"startupinfo": si, "creationflags": subprocess.CREATE_NO_WINDOW}


# ──────────────────────────────────────────────────────────────────────────────
# Service launchers
# ──────────────────────────────────────────────────────────────────────────────

def launch_backend(backend_port: int, frontend_port: int) -> subprocess.Popen:
    """
    Start uvicorn via the project venv.

    Overrides ALLOWED_ORIGINS so CORS permits the dynamically chosen
    frontend port, regardless of what is in backend/.env.
    """
    if not VENV_PYTHON.exists():
        setup_cmd = "START_APP.bat" if IS_WINDOWS else "bash start_app.sh"
        sys.exit(
            f"\n  ERROR: Python virtual environment not found.\n"
            f"  Close this window and run {setup_cmd} — it installs\n"
            f"  everything automatically on first run.\n"
        )

    env = os.environ.copy()

    # pydantic-settings reads .env for everything else;
    # we only override the two values that depend on runtime ports.
    env["ALLOWED_ORIGINS"] = json.dumps([
        f"http://localhost:{frontend_port}",
        f"http://127.0.0.1:{frontend_port}",
    ])

    return subprocess.Popen(
        [str(VENV_PYTHON), "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", str(backend_port)],
        cwd=str(BACKEND_DIR),
        env=env,
        **_popen_kwargs(),
    )


def launch_frontend(frontend_port: int, backend_port: int) -> subprocess.Popen:
    """
    Start Vite directly via node (bypasses npm.cmd shell wrapper so the
    returned PID is the actual node process, not a transient cmd.exe).

    Sets VITE_API_URL so axios calls the discovered backend port directly
    instead of relying on the Vite proxy (which has a hardcoded target).
    """
    node_exe = shutil.which("node")
    if node_exe is None:
        sys.exit("\n  ERROR: 'node' not found in PATH.  Install Node.js from https://nodejs.org\n")
    if not VITE_JS.exists():
        setup_cmd = "START_APP.bat" if IS_WINDOWS else "bash start_app.sh"
        sys.exit(
            f"\n  ERROR: Frontend dependencies not installed.\n"
            f"  Close this window and run {setup_cmd} — it runs\n"
            f"  npm install automatically on first run.\n"
        )

    env = os.environ.copy()
    # Overrides the empty VITE_API_URL in .env.development.
    # Vite's dotenv loader does NOT override an existing process env var,
    # so this value is used as-is.
    env["VITE_API_URL"] = f"http://localhost:{backend_port}"

    return subprocess.Popen(
        [node_exe, str(VITE_JS), "--port", str(frontend_port), "--host"],
        cwd=str(FRONTEND_DIR),
        env=env,
        **_popen_kwargs(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _adopt_orphan() -> dict | None:
    """
    Detect an orphaned instance — backend + frontend running on the default
    ports but with no lock file (happens when the user closes the terminal
    instead of using STOP_APP, or when stop_app cleans the lock but the
    processes survive).

    Checks the default backend port for a valid /api/v1/health response and
    the default frontend port for any HTTP 200.  If both are up, returns a
    synthetic lock dict so the caller can reuse the existing instance instead
    of spawning a duplicate on port 8001 / 5174.
    """
    b_port = BACKEND_PORT_START
    f_port = FRONTEND_PORT_START
    if http_ok(f"http://localhost:{b_port}/api/v1/health") and http_ok(f"http://localhost:{f_port}/"):
        return {"backend_port": b_port, "frontend_port": f_port,
                "backend_pid": 0, "frontend_pid": 0}
    return None


def log(label: str, value: str = "") -> None:
    if value:
        ok = value.startswith("READY") or value.startswith("Opened") or value.startswith("Loaded")
        err = value.startswith("TIMEOUT") or value.startswith("ERROR")
        val_color = GRN if ok else (RED if err else RST)
        print(f"  {CYN}{label:<22}{RST} {val_color}{value}{RST}")
    else:
        print(f"  {YLW}{label}{RST}")


def main() -> None:
    print()
    print(f"{BLD}{CYN}{SEP}{RST}")
    print(f"{BLD}{CYN}  ZnO Supercapacitor AI Platform{RST}")
    print(f"{BLD}{CYN}{SEP}{RST}")
    print()

    # ── STEP 1: Detect existing instance ──────────────────────────────────────
    existing = check_existing()

    # ── STEP 1b: Detect orphaned instance (running but no lock file) ──────────
    # When STOP_APP is skipped the processes keep running but the lock file is
    # gone.  Without this check every START_APP click spawns a new instance on
    # a higher port (8001, 8002 …) while the original keeps running.
    if not existing:
        existing = _adopt_orphan()

    if existing:
        fp = existing["frontend_port"]
        bp = existing["backend_port"]
        print(f"  {YLW}Already running — reopening browser{RST}")
        print()
        log("Backend",  f"http://localhost:{bp}")
        log("Frontend", f"http://localhost:{fp}")
        print()
        open_app_window(f"http://localhost:{fp}")
        log("Browser", "Opened")
        print()
        return

    # ── STEP 2 & 3: Find available ports ──────────────────────────────────────
    log("Scanning for free ports ...")
    backend_port  = find_free_port(BACKEND_PORT_START)
    frontend_port = find_free_port(FRONTEND_PORT_START)
    print()
    log("Backend port",  str(backend_port))
    log("Frontend port", str(frontend_port))
    print()

    # ── STEP 4: Launch backend ────────────────────────────────────────────────
    log("Starting backend ...")
    be_proc = launch_backend(backend_port, frontend_port)

    # ── STEP 5 & 6: Launch frontend with dynamic API URL ─────────────────────
    log("Starting frontend ...")
    fe_proc = launch_frontend(frontend_port, backend_port)

    # Write lock file immediately so a second double-click detects this instance
    write_lock(backend_port, frontend_port, be_proc.pid, fe_proc.pid)
    print()

    # ── STEP 7a: Wait for frontend (fast — usually < 10 s) ───────────────────
    log("Waiting for frontend ...")
    fe_url = f"http://localhost:{frontend_port}"
    fe_ok  = wait_for_url(fe_url + "/", FRONTEND_TIMEOUT, 1.5)
    log("Frontend", "READY" if fe_ok else "TIMEOUT — check frontend window")

    # ── STEP 8: Open browser as soon as frontend is up ────────────────────────
    url = f"http://localhost:{frontend_port}"
    open_app_window(url)
    log("Browser", "Opened")
    print()

    # ── STEP 7b: Wait for backend (slow — RF model ~45 s) ────────────────────
    log("Waiting for backend ML models (up to 120 s) ...")
    be_url = f"http://localhost:{backend_port}/api/v1/health"
    be_ok  = wait_for_url(be_url, BACKEND_TIMEOUT, 2.0)
    log("Backend", "READY" if be_ok else "TIMEOUT — check backend window")

    # ── STEP 9: Final summary ─────────────────────────────────────────────────
    print()
    print(f"{BLD}{GRN}{SEP}{RST}")
    log("Backend",  f"http://localhost:{backend_port}")
    log("Frontend", f"http://localhost:{frontend_port}")
    log("API",      "READY" if be_ok else "STILL LOADING — page updates automatically")
    log("Models",   "Loaded" if be_ok else "Loading in background ...")
    log("Browser",  "Opened")
    print(f"{BLD}{GRN}{SEP}{RST}")
    print()


if __name__ == "__main__":
    main()
