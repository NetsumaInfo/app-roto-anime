@echo off
echo ========================================
echo   BiRefNet Background Removal
echo ========================================
echo.

REM Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] venv not found. Run install.bat first!
    pause
    exit /b 1
)

echo Loading... Browser will open automatically.
echo Press Ctrl+C to stop
echo.

python app.py

pause
