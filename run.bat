@echo off
REM E.D.I.T.H. Vehicle Identification System - Windows Launcher
REM Usage: run.bat <image_path>

echo ========================================
echo E.D.I.T.H. Vehicle Identification System
echo ========================================
echo.

REM Check if image path is provided
if "%~1"=="" (
    echo Error: No image path provided
    echo.
    echo Usage: run.bat ^<image_path^>
    echo Example: run.bat C:\Users\YourName\Pictures\car.jpg
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Note: Virtual environment not found. Using system Python.
)

REM Check if image file exists
if not exist "%~1" (
    echo Error: Image file not found: %~1
    echo.
    pause
    exit /b 1
)

echo Processing image: %~1
echo.

REM Run the identification pipeline
python example.py --image "%~1" --verbose

echo.
echo ========================================
echo Processing complete!
echo ========================================
pause
