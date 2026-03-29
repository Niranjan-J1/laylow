#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

ROOT     = Path(__file__).parent
MODELS   = ROOT / "models"
FRONTEND = ROOT / "frontend"

BANNER = r"""
  _                _               
 | |    __ _ _  _| |___  ____ __  
 | |__ / _` | || | / _ \/ _\\ V /  
 |____|\\__,_|\\_, |_\\___/\\__/\\_/   
             |__/                  
 local AI - private by default
"""

def check_model():
    models = list(MODELS.glob("*.gguf"))
    if not models:
        print("No models found in models/ folder.")
        print()
        print("Downloading TinyLlama (608MB)...")
        MODELS.mkdir(exist_ok=True)
        try:
            import urllib.request
            url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
            dest = MODELS / "tinyllama.gguf"
            def progress(count, block, total):
                pct = int(count * block * 100 / total)
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                print(f"\r  [{bar}] {pct}%", end="", flush=True)
            urllib.request.urlretrieve(url, dest, progress)
            print("\n  Download complete.")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Manually place a .gguf file in the models/ folder.")
            sys.exit(1)

def start_api():
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--port", "8080", "--log-level", "error"],
        cwd=ROOT, env=env
    )
    return proc

def wait_for_api():
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1)
            return True
        except:
            time.sleep(1)
    return False

def start_frontend():
    if not (FRONTEND / "node_modules").exists():
        print("  Installing frontend dependencies...")
        subprocess.run(["npm.cmd", "install"], cwd=FRONTEND,
                      capture_output=True)

    proc = subprocess.Popen(
        ["npm.cmd", "run", "dev", "--", "--port", "3000"],
        cwd=FRONTEND,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc

def main():
    print(BANNER)

    # Check for model
    print("Checking models...")
    check_model()
    print("  Models ready.")
    print()

    # Start API
    print("Starting laylow server...")
    api_proc = start_api()

    if not wait_for_api():
        print("  Server failed to start.")
        api_proc.terminate()
        sys.exit(1)
    print("  Server running on http://127.0.0.1:8080")
    print()

    # Start frontend
    print("Starting chat UI...")
    fe_proc = start_frontend()
    time.sleep(3)
    print("  UI running on http://localhost:3000")
    print()

    # Open browser
    print("Opening browser...")
    webbrowser.open("http://localhost:3000")
    print()
    print("laylow is running. Press Ctrl+C to stop.")
    print()

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down laylow...")
        api_proc.terminate()
        fe_proc.terminate()
        print("Goodbye.")

if __name__ == "__main__":
    main()