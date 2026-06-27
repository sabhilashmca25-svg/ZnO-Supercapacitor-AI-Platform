"""
ZnO Supercapacitor AI Platform — Smart Stopper

Reads .launcher.lock and terminates ONLY the backend and frontend
processes that belong to this project.  Never touches other projects.

Cross-platform: works on Windows, macOS, and Linux.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
LOCK_FILE = Path(__file__).resolve().parent / ".launcher.lock"
SEP = "-" * 54


def is_pid_alive(pid: int) -> bool:
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


def kill_tree(pid: int) -> bool:
    if IS_WINDOWS:
        r = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, text=True,
        )
        return r.returncode == 0
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.8)
            if is_pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return True  # already dead
        except Exception:
            return False


def main() -> None:
    print()
    print(SEP)
    print("  ZnO Supercapacitor AI Platform -- Stop")
    print(SEP)
    print()

    if not LOCK_FILE.exists():
        print("  Not running (no lock file found).")
        print()
        if IS_WINDOWS:
            input("  Press Enter to close ...")
        return

    try:
        data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ERROR: Cannot read lock file: {e}")
        print()
        if IS_WINDOWS:
            input("  Press Enter to close ...")
        return

    backend_pid   = data.get("backend_pid")
    frontend_pid  = data.get("frontend_pid")
    backend_port  = data.get("backend_port")
    frontend_port = data.get("frontend_port")

    stopped_any = False

    if backend_pid:
        if is_pid_alive(backend_pid):
            ok = kill_tree(backend_pid)
            status = "stopped" if ok else "ERROR stopping"
        else:
            status = "already stopped"
        print(f"  Backend  PID {backend_pid:<6}  port {backend_port}  ->  {status}")
        stopped_any = True

    if frontend_pid:
        if is_pid_alive(frontend_pid):
            ok = kill_tree(frontend_pid)
            status = "stopped" if ok else "ERROR stopping"
        else:
            status = "already stopped"
        print(f"  Frontend PID {frontend_pid:<6}  port {frontend_port}  ->  {status}")
        stopped_any = True

    LOCK_FILE.unlink(missing_ok=True)

    print()
    if stopped_any:
        print("  All project processes terminated.")
    else:
        print("  Nothing to stop — processes were already gone.")
    print("  Lock file removed.")
    print()
    if IS_WINDOWS:
        input("  Press Enter to close ...")


if __name__ == "__main__":
    main()
