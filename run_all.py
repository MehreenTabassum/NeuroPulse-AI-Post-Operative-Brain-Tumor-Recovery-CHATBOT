"""
run_all.py — Single launcher for the entire NeuroRecovery AI stack.

Starts three processes in order:
  1. vision_service.py  → http://localhost:8001  (MRI feature extraction)
  2. main.py            → http://localhost:8000  (LangGraph agent API)
  3. Next.js frontend   → http://localhost:3000  (dashboard)

Usage:
    python run_all.py

Stop: Ctrl+C  (kills all child processes cleanly)
"""

import subprocess
import sys
import time
import signal
import os

# On Windows, npm/npx are .cmd batch files — they need shell=True to resolve.
IS_WINDOWS = sys.platform == "win32"

SERVICES = [
    {
        "name": "Vision Service (port 8001)",
        # reload=False: torch multiprocessing conflicts with uvicorn file-watcher
        "cmd": [sys.executable, "-m", "uvicorn", "vision_service:app",
                "--host", "0.0.0.0", "--port", "8001"],
        "shell": False,
        "cwd": ".",
    },
    {
        "name": "Agent API (port 8000)",
        "cmd": [sys.executable, "-m", "uvicorn", "main:app",
                "--host", "0.0.0.0", "--port", "8000", "--reload"],
        "shell": False,
        "cwd": ".",
    },
    {
        "name": "Next.js Frontend (port 3000)",
        # Pass as a string + shell=True so Windows finds npm.cmd automatically.
        "cmd": "npm run dev",
        "shell": True,
        "cwd": "./frontend",    # ← rename if your Next.js folder is named differently
    },
]

procs = []

def shutdown(sig=None, frame=None):
    print("\n[run_all] Shutting down all services...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

for svc in SERVICES:
    print(f"[run_all] Starting → {svc['name']}")
    p = subprocess.Popen(
        svc["cmd"],
        cwd=svc["cwd"],
        shell=svc["shell"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    procs.append(p)
    time.sleep(10)   # stagger so each service binds its port before the next starts

print("\n[run_all] All services running.")
print("  Vision API : http://localhost:8001/docs")
print("  Agent  API : http://localhost:8000/docs")
print("  Frontend   : http://localhost:3000")
print("\nPress Ctrl+C to stop everything.\n")

# Keep alive — exit if any child dies unexpectedly
while True:
    for p in procs:
        if p.poll() is not None:
            print(f"[run_all] A service exited unexpectedly (pid {p.pid}). Shutting down.")
            shutdown()
    time.sleep(3)
