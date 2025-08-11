import os, sys, subprocess, venv, platform, signal
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV_DIR = ROOT / ".venv"
PYTHON = VENV_DIR / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

def ensure_venv():
    if not VENV_DIR.exists():
        print("[start] Creating virtual environment at", VENV_DIR)
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print("[start] Using existing virtual environment:", VENV_DIR)

def pip_install():
    req = ROOT / "requirements.txt"
    if not req.exists():
        print("[start] No requirements.txt found; skipping install")
        return
    print("[start] Installing requirements from", req)
    subprocess.check_call([str(PYTHON), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    subprocess.check_call([str(PYTHON), "-m", "pip", "install", "-r", str(req)])

def run_script(script_rel):
    target = (ROOT / script_rel).resolve()
    if not target.exists():
        raise FileNotFoundError(f"[start] Cannot find script: {target}")

    is_windows = platform.system() == "Windows"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")  # flush child logs immediately

    print(f"[start] Python: {PYTHON}")
    print(f"[start] CWD:    {ROOT}")
    print(f"[start] Script: {target}")
    print("[start] Launching... (Ctrl+C to stop)")

    if is_windows:
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        proc = subprocess.Popen(
            [str(PYTHON), "-u", str(target)],
            cwd=str(ROOT), env=env,
            creationflags=CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            [str(PYTHON), "-u", str(target)],
            cwd=str(ROOT), env=env,
            preexec_fn=os.setsid,  # new process group so we can signal it
        )

    try:
        rc = proc.wait()
        print(f"[start] Child exited with return code {rc}")
        sys.exit(rc)
    except KeyboardInterrupt:
        print("[start] Stopping app...")
        try:
            if is_windows:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGINT)
            rc = proc.wait(timeout=5)
            print(f"[start] Stopped (SIGINT), rc={rc}")
            sys.exit(rc)
        except Exception:
            try:
                if is_windows:
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                rc = proc.wait(timeout=3)
                print(f"[start] Stopped (SIGTERM), rc={rc}")
                sys.exit(rc)
            except Exception:
                if is_windows:
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                print("[start] Killed process group")
                sys.exit(1)

if __name__ == "__main__":
    script_rel = sys.argv[1] if len(sys.argv) > 1 else "scripts/mean_variance_optimal_portfolio.py"
    ensure_venv()
    pip_install()
    run_script(script_rel)
