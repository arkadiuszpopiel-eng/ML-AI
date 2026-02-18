#!/usr/bin/env python3
"""
NeuroForge - Launch Script
Starts the FastAPI server and opens the browser.
"""
import os
import sys
import logging
import webbrowser
import threading

# Ensure we're running from the right directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Add project to path
sys.path.insert(0, SCRIPT_DIR)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def open_browser(port):
    """Open browser after a short delay."""
    def _open():
        import time
        time.sleep(2)
        url = f"http://localhost:{port}"
        print(f"\n  Opening browser at: {url}")
        webbrowser.open(url)
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    setup_logging()

    print()
    print("  NeuroForge - Local AI Agent Studio")
    print("  ===================================")
    print()

    # Load config
    try:
        import yaml
        config_path = os.path.join(SCRIPT_DIR, "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}

    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 7860)

    print(f"  Server: http://{host}:{port}")
    print(f"  Local:  http://localhost:{port}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    # Open browser
    open_browser(port)

    # Start server
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
