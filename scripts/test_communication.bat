@echo off
echo ========================================
echo Super Mario Bros AI Communication Test
echo ========================================
echo.
echo This script tests WebSocket communication between
echo the Python trainer and FCEUX Lua script.
echo.
echo INSTRUCTIONS:
echo 1. Start FCEUX with Super Mario Bros ROM
echo 2. Load the Lua script: lua/mario_ai.lua
echo 3. Press any key to start the test
echo.
pause

echo Starting communication test...
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Run the communication test
python test_communication.py --duration 30

echo.
echo Test completed. Check logs/communication_test.log for details.
echo.
pause