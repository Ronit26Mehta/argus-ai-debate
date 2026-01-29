@echo off
REM BioSage Terminal - Build and Publish Script
REM This script automates the build and publish process

echo ========================================
echo BioSage Terminal - Build and Publish
echo ========================================
echo.

REM Navigate to project directory
cd /d "%~dp0"

echo [Step 1] Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist biosage_terminal.egg-info rmdir /s /q biosage_terminal.egg-info
echo Previous builds cleaned.
echo.

echo [Step 2] Building package...
python -m build
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo Build completed successfully.
echo.

echo [Step 3] Checking distribution...
twine check dist/*
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Distribution check failed!
    pause
    exit /b 1
)
echo Distribution check passed.
echo.

echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Files created in 'dist' folder:
dir dist
echo.

echo Next steps:
echo   1. Test upload: twine upload --repository testpypi dist/*
echo   2. Production upload: twine upload dist/*
echo.

pause
