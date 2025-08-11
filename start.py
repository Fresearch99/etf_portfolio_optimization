"""
start.py — one-click bootstrapper

What it does:
1) Ensures a local virtual environment exists at ./.venv
2) Installs dependencies from requirements.txt (if present)
3) Launches the target Python script as a child process
4) Forwards Ctrl+C cleanly to the child across macOS/Linux/Windows

Usage:
    python start.py scripts/mean_variance_optimal_portfolio.py
    # or with a different script:
    python start.py scripts/efficient_frontier.py
"""

import os
import sys
import signal
import venv
import platform
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------

# Repository root = directory containing this start.py
ROOT: Path = Path(__file__).parent.resolve()

# Virtual environment directory and its Python executable (platform-specific paths)
VENV_DIR: Path = ROOT / ".venv"
PYTHON: Path = VENV_DIR / (
    "Scripts/python.exe" if platform.system() == "Windows" else "bin/python"
)


# --------------------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------------------

def ensure_venv() -> None:
    """
    Create a virtual environment under ./.venv if it doesn't exist yet.
    Leaves an existing environment untouched.
    """
    if not VENV_DIR.exists():
        print("[start] Creating virtual environment at", VENV_DIR)
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print("[start] Using existing virtual environment:", VENV_DIR)


def pip_install() -> None:
    """
    Install/upgrade dependencies from requirements.txt into the local venv.
    If no requirements.txt is present, skip silently.
    """
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("[start] No requirements.txt found; skipping install")
        return

    print("[start] Installing requirements from", req)
    # Keep pip/setuptools/wheel fresh to reduce install hiccups
    subprocess.check_call(
        [str(PYTHON), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"]
    )
    # Install repo dependencies
    subprocess.check_call([str(PYTHON), "-m", "pip", "install", "-r", str(req)])


# --------------------------------------------------------------------------------------
# Process launching & signal forwarding
# --------------------------------------------------------------------------------------

def run_script(script_rel: str) -> None:
    """
    Launch the target script as a child process and forward Ctrl+C to it.

    - On Windows, the child is started in a NEW PROCESS GROUP so we can send
      CTRL_BREAK_EVENT to that group.
    - On macOS/Linux, the child starts in a NEW SESSION (own process group)
      so we can send SIGINT/SIGTERM/SIGKILL to the whole group.
    """
    # Resolve target script path (relative to repo root)
    target = (ROOT / script_rel).resolve()
    if not target.exists():
        raise FileNotFoundError(f"[start] Cannot find script: {target}")

    is_windows = platform.system() == "Windows"

    # Inherit the parent env; ensure unbuffered output for real-time logs
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"[start] Python: {PYTHON}")
    print(f"[start] CWD:    {ROOT}")
    print(f"[start] Script: {target}")
    print("[start] Launching... (Ctrl+C to stop)")

    # Start child process in its own group/session so we can signal it reliably
    if is_windows:
        # CREATE_NEW_PROCESS_GROUP lets us later send CTRL_BREAK_EVENT
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        proc = subprocess.Popen(
            [str(PYTHON), "-u", str(target)],
            cwd=str(ROOT),
            env=env,
            creationflags=CREATE_NEW_PROCESS_GROUP,
        )
    else:
        # preexec_fn=os.setsid -> child runs in a new session (own process group)
        proc = subprocess.Popen(
            [str(PYTHON), "-u", str(target)],
            cwd=str(ROOT),
            env=env,
            preexec_fn=os.setsid,
        )

    # Wait for child to finish, or handle Ctrl+C to stop it gracefully
    try:
        rc = proc.wait()
        print(f"[start] Child exited with return code {rc}")
        sys.exit(rc)

    except KeyboardInterrupt:
        print("[start] Stopping app...")

        # 1) Try a gentle interrupt first
        try:
            if is_windows:
                # Send Ctrl+Break to the child's process group
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Send SIGINT to the child's process group
                os.killpg(proc.pid, signal.SIGINT)

            rc = proc.wait(timeout=5)
            print(f"[start] Stopped (SIGINT), rc={rc}")
            sys.exit(rc)

        # 2) If it didn't stop, try terminate (SIGTERM / terminate())
        except Exception:
            try:
                if is_windows:
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)

                rc = proc.wait(timeout=3)
                print(f"[start] Stopped (SIGTERM), rc={rc}")
                sys.exit(rc)

            # 3) Last resort: hard kill
            except Exception:
                if is_windows:
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)

                print("[start] Killed process group")
                sys.exit(1)


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Default target if none provided on the command line
    script_rel = sys.argv[1] if len(sys.argv) > 1 else "scripts/mean_variance_optimal_portfolio.py"

    ensure_venv()
    pip_install()
    run_script(script_rel)

