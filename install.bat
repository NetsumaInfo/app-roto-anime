@echo off
echo ========================================
echo   BiRefNet Background Removal - Install
echo ========================================
echo.

REM Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Download from: https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during install!
    pause
    exit /b 1
)
echo       Python OK

REM Check ffmpeg
echo [2/4] Checking ffmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ffmpeg not found - video processing won't work
    echo To install: winget install ffmpeg
    echo.
) else (
    echo       ffmpeg OK
)

REM Create venv
echo [3/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)

REM Install dependencies
echo [4/4] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM Install torch first (important for torchvision compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
pip install transformers gradio Pillow opencv-python huggingface-hub tqdm timm kornia einops "numpy<2"

echo.
echo ========================================
echo   Installation complete!
echo   Run 'run.bat' to start the app
echo ========================================

REM Check for ToonOut weights
if not exist "weights\birefnet_finetuned_toonout.pth" (
    echo.
    echo [INFO] ToonOut weights not found (optional)
    echo Download from: https://huggingface.co/joelseytre/toonout
)

pause
