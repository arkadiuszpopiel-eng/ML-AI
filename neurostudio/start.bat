@echo off
REM ============================================================
REM  NeuroForge - Local AI Agent Studio
REM  All-in-one: install + run
REM  Usage: double-click start.bat or run from cmd
REM ============================================================
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_DIR=%SCRIPT_DIR%\venv"
set "BIN_DIR=%SCRIPT_DIR%\bin"
set "MODELS_DIR=%SCRIPT_DIR%\models"
set "DATA_DIR=%SCRIPT_DIR%\data"
set "REQ_FILE=%SCRIPT_DIR%\requirements.txt"

echo ============================================================
echo   NeuroForge - Local AI Agent Studio v0.7.0
echo ============================================================
echo.

REM ─── Step 1: Check Python ───
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Python is not installed or not in PATH.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Found Python %PYVER%

REM ─── Step 2: Create/check virtual environment ───
echo [2/5] Checking virtual environment...
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   Virtual environment exists.
) else (
    echo   Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo   ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   Virtual environment created.
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

REM ─── Step 3: Install/update Python packages ───
echo [3/5] Checking Python packages...

REM Use a stamp file to avoid re-installing every time
set "STAMP_FILE=%VENV_DIR%\.requirements_stamp"
set "NEEDS_INSTALL=0"

if not exist "%STAMP_FILE%" set "NEEDS_INSTALL=1"

REM Check if requirements.txt is newer than stamp
if exist "%STAMP_FILE%" (
    for %%A in ("%REQ_FILE%") do set REQ_TIME=%%~tA
    for %%A in ("%STAMP_FILE%") do set STAMP_TIME=%%~tA
    if "!REQ_TIME!" neq "!STAMP_TIME!" set "NEEDS_INSTALL=1"
)

if "!NEEDS_INSTALL!"=="1" (
    echo   Installing packages...
    "%VENV_PIP%" install --upgrade pip >nul 2>&1
    "%VENV_PIP%" install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo   ERROR: Failed to install packages.
        pause
        exit /b 1
    )
    copy /y "%REQ_FILE%" "%STAMP_FILE%" >nul 2>&1
    echo   Packages installed.
) else (
    echo   Packages up to date.
)

REM ─── Step 4: Download llama.cpp if not present ───
echo [4/5] Checking llama.cpp binary...
set "LLAMA_FOUND=0"

if exist "%BIN_DIR%" (
    for /r "%BIN_DIR%" %%f in (llama-server.exe) do (
        set "LLAMA_PATH=%%f"
        set "LLAMA_FOUND=1"
    )
)

if "!LLAMA_FOUND!"=="1" (
    echo   llama-server found: !LLAMA_PATH!
) else (
    echo   llama-server not found. Downloading...
    "%VENV_PYTHON%" -c "import sys; sys.path.insert(0, r'%SCRIPT_DIR%'); from install import step_download_llama_cpp; step_download_llama_cpp(4, 5)"
    if errorlevel 1 (
        echo   WARNING: Could not download llama.cpp. You can download it manually.
        echo   See: https://github.com/ggml-org/llama.cpp/releases
    )
)

REM ─── Step 5: Create directories ───
echo [5/5] Checking directories...
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%DATA_DIR%\conversations" mkdir "%DATA_DIR%\conversations"
if not exist "%DATA_DIR%\documents" mkdir "%DATA_DIR%\documents"
if not exist "%DATA_DIR%\uploads" mkdir "%DATA_DIR%\uploads"
echo   Directories OK.

REM ─── Launch ───
echo.
echo ============================================================
echo   Starting NeuroForge server...
echo   Web UI: http://localhost:7860
echo   Press Ctrl+C to stop.
echo ============================================================
echo.

cd /d "%SCRIPT_DIR%"
"%VENV_PYTHON%" run.py

if errorlevel 1 (
    echo.
    echo   Server stopped with an error.
)

echo.
pause
