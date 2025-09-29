@echo off
echo ================================================================
echo  Super Mario Bros AI Training System - Triton Installation Fix
echo ================================================================
echo.

echo Activating virtual environment...
call mario_ai_env\Scripts\activate.bat

echo.
echo Fixing Triton installation for PyTorch compilation...
echo.

echo Step 1: Uninstalling existing Triton (if present)...
pip uninstall triton -y

echo.
echo Step 2: Installing compatible Triton version...
pip install triton>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 3: Running Triton fix script...
python fix_triton.py

echo.
echo Step 4: Updating requirements...
pip install -r requirements.txt

echo.
echo ================================================================
echo  Triton Installation Fix Complete
echo ================================================================
echo.
echo The Triton installation has been fixed. You can now run the
echo training system without the compilation errors.
echo.
echo Next steps:
echo 1. Close this window
echo 2. Run the training system again
echo 3. The Triton errors should be resolved
echo.
pause