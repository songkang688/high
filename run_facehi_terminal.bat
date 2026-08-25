@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "ROOT=%~dp0"

set "PYTHONHOME="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_PYTHON_EXE="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CUDA_VISIBLE_DEVICES="
set "NVIDIA_VISIBLE_DEVICES="

set "PY=%ROOT%python\python.exe"
set "APP_DIR=%ROOT%main"
set "CLI=%APP_DIR%\cli_process.py"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if "%~1"=="" goto :open_terminal

if not exist "%PY%" (
    echo [FAIL] Missing bundled python:
    echo   %PY%
    echo Run this from the installed app folder, not the source code folder.
    exit /b 11
)
if not exist "%CLI%" (
    echo [FAIL] Missing CLI script:
    echo   %CLI%
    exit /b 12
)

set "PYTHONPATH=%APP_DIR%"
"%PY%" "%CLI%" %*
exit /b %ERRORLEVEL%

:open_terminal
if not exist "%ROOT%open_facehi_terminal.ps1" (
    echo [FAIL] Missing launcher script:
    echo   %ROOT%open_facehi_terminal.ps1
    exit /b 13
)
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%ROOT%open_facehi_terminal.ps1"
exit /b %ERRORLEVEL%
