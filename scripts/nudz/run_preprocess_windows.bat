@echo off
setlocal

REM Wrapper for scripts/nudz/run_preprocess_windows.ps1
REM Usage:
REM   run_preprocess_windows.bat            -> full mode
REM   run_preprocess_windows.bat full       -> full mode
REM   run_preprocess_windows.bat smoke      -> smoke mode

set MODE=%~1
if "%MODE%"=="" set MODE=full

cd /d "%~dp0\..\.."

powershell -ExecutionPolicy Bypass -File "scripts\nudz\run_preprocess_windows.ps1" -Mode %MODE%
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo Preprocessing failed with exit code %EXITCODE%.
    pause
    exit /b %EXITCODE%
)

echo.
echo Preprocessing finished successfully.
pause
exit /b 0
