@echo off
echo ========================================
echo   BiRefNet Background Removal - Install
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)

echo [2/3] Activating venv and installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Checking for ToonOut weights...
if not exist "weights\birefnet_finetuned_toonout.pth" (
    echo.
    echo [INFO] ToonOut weights not found.
    echo Download from: https://huggingface.co/joelseytre/toonout
    echo Place in: weights\birefnet_finetuned_toonout.pth
)

echo.
echo ========================================
echo   Installation complete!
echo   Run 'run.bat' to start the app
echo ========================================
pause
