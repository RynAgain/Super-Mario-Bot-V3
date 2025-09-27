@echo off
REM Super Mario Bros AI Training System - Windows Installation Script
REM This script automates the installation process on Windows systems

echo ================================================================
echo  Super Mario Bros AI Training System - Windows Installation
echo ================================================================
echo.

REM Check if Python is installed
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
echo [2/8] Verifying Python version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

REM Check if version is 3.8 or higher
if %MAJOR% lss 3 (
    echo ERROR: Python 3.8 or higher is required. Found Python %PYTHON_VERSION%
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 8 (
    echo ERROR: Python 3.8 or higher is required. Found Python %PYTHON_VERSION%
    pause
    exit /b 1
)

echo Python version check passed!

REM Check if pip is available
echo [3/8] Checking pip installation...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed or not in PATH
    echo Please install pip or reinstall Python with pip included
    pause
    exit /b 1
)
echo pip is available!

REM Upgrade pip to latest version
echo [4/8] Upgrading pip to latest version...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip, continuing with current version
)

REM Create virtual environment (optional but recommended)
echo [5/8] Setting up virtual environment...
set /p CREATE_VENV="Create virtual environment? (recommended) [Y/n]: "
if /i "%CREATE_VENV%"=="n" goto skip_venv
if /i "%CREATE_VENV%"=="no" goto skip_venv

echo Creating virtual environment 'mario_ai_env'...
python -m venv mario_ai_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call mario_ai_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment created and activated!
goto install_deps

:skip_venv
echo Skipping virtual environment creation...

:install_deps
REM Install dependencies
echo [6/8] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Install the package in development mode
echo [7/8] Installing Super Mario AI package...
pip install -e .
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Super Mario AI package
    pause
    exit /b 1
)

REM Run system validation
echo [8/8] Running system validation...
python validate_system.py
if %errorlevel% neq 0 (
    echo WARNING: System validation failed
    echo The installation completed but there may be issues
    echo Please check the error messages above
) else (
    echo System validation passed!
)

echo.
echo ================================================================
echo  Installation Complete!
echo ================================================================
echo.
echo Next Steps:
echo 1. Download and install FCEUX emulator from http://fceux.com
echo 2. Load Super Mario Bros ROM in FCEUX
echo 3. Load the Lua script: lua/mario_ai.lua
echo 4. Run training: run_training.bat
echo.
echo For detailed instructions, see:
echo - README.md for overview
echo - INSTALLATION.md for setup guide  
echo - USAGE.md for usage examples
echo.
echo Troubleshooting:
echo - If you encounter issues, see TROUBLESHOOTING.md
echo - Run tests: python test_complete_system_integration.py
echo - Validate system: python validate_system.py
echo.

REM Check if virtual environment was created
if exist "mario_ai_env" (
    echo Virtual Environment:
    echo - To activate: mario_ai_env\Scripts\activate.bat
    echo - To deactivate: deactivate
    echo.
)

echo Happy training! 🎮🚀
echo.
pause