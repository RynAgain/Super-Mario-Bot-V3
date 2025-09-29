@echo off
REM Super Mario Bros AI Training System - Training Startup Script
REM This script provides an easy way to start training with various options

echo ================================================================
echo  Super Mario Bros AI Training System - Training Startup
echo ================================================================
echo.

REM Check if virtual environment exists and activate it
if exist "mario_ai_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call mario_ai_env\Scripts\activate.bat
    echo Virtual environment activated!
    echo.
    set PYTHON_CMD=mario_ai_env\Scripts\python.exe
) else (
    echo No virtual environment found, using global Python...
    set PYTHON_CMD=python
    echo.
)

REM Check if Python and required packages are available
echo Checking system requirements...
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not available
    echo Please run install.bat first or ensure Python is in your PATH
    pause
    exit /b 1
)

REM Check if the main module is available
if exist "mario_ai_env\Scripts\python.exe" (
    mario_ai_env\Scripts\python.exe -c "import python.main" >nul 2>&1
) else (
    %PYTHON_CMD% -c "import sys; sys.path.insert(0, '.'); import python.main" >nul 2>&1
)
if %errorlevel% neq 0 (
    echo ERROR: Super Mario AI package not found
    echo Please run install.bat first to install the package
    pause
    exit /b 1
)

echo System requirements check passed!
echo.

REM Display menu options
:menu
echo ================================================================
echo  Training Options
echo ================================================================
echo.
echo 1. Quick Start Training (default settings)
echo 2. Custom Configuration Training
echo 3. Resume Training from Checkpoint
echo 4. Run System Tests
echo 5. Validate System
echo 6. View Training Logs
echo 7. Generate Performance Plots
echo 8. Exit
echo.
set /p choice="Select an option [1-8]: "

if "%choice%"=="1" goto quick_start
if "%choice%"=="2" goto custom_config
if "%choice%"=="3" goto resume_training
if "%choice%"=="4" goto run_tests
if "%choice%"=="5" goto validate_system
if "%choice%"=="6" goto view_logs
if "%choice%"=="7" goto generate_plots
if "%choice%"=="8" goto exit
echo Invalid choice. Please select 1-8.
goto menu

:quick_start
echo.
echo ================================================================
echo  Quick Start Training
echo ================================================================
echo.
echo Starting training with default configuration...
echo.
echo IMPORTANT: Make sure FCEUX is running with the Lua script loaded!
echo - Load Super Mario Bros ROM in FCEUX
echo - Load lua/mario_ai.lua script in FCEUX
echo - The script should show "Waiting for connection..." message
echo.
set /p continue="Press Enter when FCEUX is ready, or 'q' to quit: "
if /i "%continue%"=="q" goto menu

echo Starting training...
%PYTHON_CMD% python/main.py train
goto post_training

:custom_config
echo.
echo ================================================================
echo  Custom Configuration Training
echo ================================================================
echo.
echo Available configuration files:
if exist "config\training_config.yaml" echo - config\training_config.yaml (default)
if exist "examples\basic_training.yaml" echo - examples\basic_training.yaml (basic)
if exist "examples\advanced_training.yaml" echo - examples\advanced_training.yaml (advanced)
echo.
set /p config_file="Enter config file path (or press Enter for default): "
if "%config_file%"=="" set config_file=config\training_config.yaml

if not exist "%config_file%" (
    echo ERROR: Configuration file not found: %config_file%
    pause
    goto menu
)

echo.
echo IMPORTANT: Make sure FCEUX is running with the Lua script loaded!
set /p continue="Press Enter when FCEUX is ready, or 'q' to quit: "
if /i "%continue%"=="q" goto menu

echo Starting training with config: %config_file%
%PYTHON_CMD% python/main.py train --config "%config_file%"
goto post_training

:resume_training
echo.
echo ================================================================
echo  Resume Training from Checkpoint
echo ================================================================
echo.
echo Available checkpoints:
if exist "checkpoints" (
    dir /b checkpoints\*.pth 2>nul
    if %errorlevel% neq 0 (
        echo No checkpoint files found in checkpoints directory
        pause
        goto menu
    )
) else (
    echo No checkpoints directory found
    pause
    goto menu
)
echo.
set /p checkpoint="Enter checkpoint filename (without path): "
if "%checkpoint%"=="" (
    echo No checkpoint specified
    pause
    goto menu
)

if not exist "checkpoints\%checkpoint%" (
    echo ERROR: Checkpoint file not found: checkpoints\%checkpoint%
    pause
    goto menu
)

echo.
echo IMPORTANT: Make sure FCEUX is running with the Lua script loaded!
set /p continue="Press Enter when FCEUX is ready, or 'q' to quit: "
if /i "%continue%"=="q" goto menu

echo Resuming training from checkpoint: %checkpoint%
%PYTHON_CMD% python/main.py train --resume "checkpoints\%checkpoint%"
goto post_training

:run_tests
echo.
echo ================================================================
echo  Running System Tests
echo ================================================================
echo.
echo Running comprehensive integration tests...
%PYTHON_CMD% test_complete_system_integration.py
echo.
echo Running neural network component tests...
%PYTHON_CMD% test_neural_network_components.py
echo.
echo Running communication system tests...
%PYTHON_CMD% test_communication_system.py
echo.
echo All tests completed!
pause
goto menu

:validate_system
echo.
echo ================================================================
echo  System Validation
echo ================================================================
echo.
echo Running system validation...
%PYTHON_CMD% validate_system.py
pause
goto menu

:view_logs
echo.
echo ================================================================
echo  Training Logs
echo ================================================================
echo.
if exist "logs" (
    echo Available log directories:
    dir /b logs 2>nul
    echo.
    set /p session="Enter session ID to view (or press Enter to list all): "
    if "%session%"=="" (
        echo Listing all log files:
        dir /s logs\*.csv 2>nul
    ) else (
        if exist "logs\%session%" (
            echo Log files for session %session%:
            dir /b logs\%session%\*.csv 2>nul
        ) else (
            echo Session directory not found: logs\%session%
        )
    )
) else (
    echo No logs directory found. Run training first to generate logs.
)
pause
goto menu

:generate_plots
echo.
echo ================================================================
echo  Generate Performance Plots
echo ================================================================
echo.
if exist "logs" (
    echo Available log sessions:
    dir /b logs 2>nul
    echo.
    set /p session="Enter session ID for plotting: "
    if "%session%"=="" (
        echo No session specified
        pause
        goto menu
    )
    
    if exist "logs\%session%" (
        echo Generating plots for session: %session%
        %PYTHON_CMD% python/mario_logging/plotter.py --session "%session%"
    ) else (
        echo Session directory not found: logs\%session%
    )
) else (
    echo No logs directory found. Run training first to generate logs.
)
pause
goto menu

:post_training
echo.
echo ================================================================
echo  Training Session Complete
echo ================================================================
echo.
echo Training session has ended.
echo.
echo You can now:
echo - View logs in the logs/ directory
echo - Generate performance plots
echo - Resume training from the latest checkpoint
echo.
set /p next_action="Return to menu? [Y/n]: "
if /i "%next_action%"=="n" goto exit
goto menu

:exit
echo.
echo ================================================================
echo  Goodbye!
echo ================================================================
echo.
echo Thank you for using Super Mario Bros AI Training System!
echo.
echo For support and documentation:
echo - README.md for overview
echo - USAGE.md for detailed usage
echo - TROUBLESHOOTING.md for common issues
echo.
if exist "mario_ai_env\Scripts\activate.bat" (
    echo Remember to deactivate the virtual environment when done:
    echo   deactivate
    echo.
)
pause