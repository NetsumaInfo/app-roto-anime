@echo off
echo ========================================
echo   BiRefNet Background Removal - Install
echo ========================================
echo.

REM Check Python
echo [1/5] Checking Python...
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
echo [2/5] Checking ffmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ffmpeg not found - video processing won't work
    echo To install: winget install ffmpeg
    echo.
) else (
    echo       ffmpeg OK
)

REM Create venv
echo [3/5] Creating virtual environment...
if exist venv (
    echo       venv already exists, skipping...
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
)

REM Install dependencies
echo [4/5] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo Installing AI libraries...
pip install transformers huggingface-hub timm kornia einops

echo Installing utilities...
pip install numpy"<2" Pillow opencv-python tqdm

echo Installing Gradio interface...
pip install gradio

REM Download ToonOut weights
echo.
echo [5/5] Downloading ToonOut weights (885 MB)...
if not exist "weights" mkdir weights
if not exist "weights\birefnet_finetuned_toonout.pth" (
    echo       Downloading from HuggingFace...
    curl -L -o "weights\birefnet_finetuned_toonout.pth" "https://huggingface.co/joelseytre/toonout/resolve/main/birefnet_finetuned_toonout.pth"
    if %errorlevel% neq 0 (
        echo [WARNING] Failed to download ToonOut weights
        echo You can download manually from: https://huggingface.co/joelseytre/toonout
    ) else (
        echo       ToonOut weights OK
    )
) else (
    echo       ToonOut weights already exist, skipping...
)

echo.
echo ========================================
echo   Installation complete!
echo   Run 'run.bat' to start the app
echo ========================================

pause
