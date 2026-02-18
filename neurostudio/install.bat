@echo off
REM ============================================================
REM  NeuroForge - One-Click Installer for Windows
REM  Installs everything automatically.
REM ============================================================

echo ============================================================
echo   NeuroForge - One-Click Installer
echo   Local AI Agent Studio
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.

REM Run installer
python "%~dp0install.py"
if errorlevel 1 (
    echo.
    echo Installation encountered errors. Check the output above.
    pause
    exit /b 1
)

echo.
pause
