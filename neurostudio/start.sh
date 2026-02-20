#!/bin/bash
# ============================================================
#  NeuroForge - Local AI Agent Studio
#  All-in-one: install + run
#  Usage: chmod +x start.sh && ./start.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
BIN_DIR="$SCRIPT_DIR/bin"
MODELS_DIR="$SCRIPT_DIR/models"
DATA_DIR="$SCRIPT_DIR/data"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

echo "============================================================"
echo "  NeuroForge - Local AI Agent Studio v0.6.0"
echo "============================================================"
echo ""

# ─── Step 1: Check Python ───
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: Python 3 is not installed."
    echo "  Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PYVER=$(python3 --version 2>&1)
echo "  Found $PYVER"

# ─── Step 2: Create/check virtual environment ───
echo "[2/5] Checking virtual environment..."
if [ -f "$VENV_DIR/bin/python" ]; then
    echo "  Virtual environment exists."
else
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "  Virtual environment created."
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ─── Step 3: Install/update Python packages ───
echo "[3/5] Checking Python packages..."
STAMP_FILE="$VENV_DIR/.requirements_stamp"
NEEDS_INSTALL=0

if [ ! -f "$STAMP_FILE" ]; then
    NEEDS_INSTALL=1
elif [ "$REQ_FILE" -nt "$STAMP_FILE" ]; then
    NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
    echo "  Installing packages..."
    "$VENV_PIP" install --upgrade pip > /dev/null 2>&1
    "$VENV_PIP" install -r "$REQ_FILE"
    cp "$REQ_FILE" "$STAMP_FILE"
    echo "  Packages installed."
else
    echo "  Packages up to date."
fi

# ─── Step 4: Download llama.cpp if not present ───
echo "[4/5] Checking llama.cpp binary..."
LLAMA_PATH=$(find "$BIN_DIR" -name "llama-server" -type f 2>/dev/null | head -1)

if [ -n "$LLAMA_PATH" ]; then
    echo "  llama-server found: $LLAMA_PATH"
else
    echo "  llama-server not found. Downloading..."
    "$VENV_PYTHON" "$SCRIPT_DIR/install.py" || {
        echo "  WARNING: Could not download llama.cpp automatically."
        echo "  Download manually from: https://github.com/ggml-org/llama.cpp/releases"
        echo "  Place the binary in: $BIN_DIR"
    }
fi

# ─── Step 5: Create directories ───
echo "[5/5] Checking directories..."
mkdir -p "$MODELS_DIR" "$DATA_DIR/conversations" "$DATA_DIR/documents" "$DATA_DIR/uploads"
echo "  Directories OK."

# ─── Launch ───
echo ""
echo "============================================================"
echo "  Starting NeuroForge server..."
echo "  Web UI: http://localhost:7860"
echo "  Press Ctrl+C to stop."
echo "============================================================"
echo ""

cd "$SCRIPT_DIR"
"$VENV_PYTHON" run.py
