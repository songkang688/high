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
set "LOG=%ROOT%startup.log"
set "ERR=%ROOT%startup_error.log"

if not exist "%PY%" (
    echo python.exe missing: %PY%>>"%ERR%"
    exit /b 11
)
if not exist "%APP_DIR%\app.py" (
    echo app.py missing: %APP_DIR%\app.py>>"%ERR%"
    exit /b 12
)

set "PYTHONPATH=%APP_DIR%"
cd /d "%APP_DIR%"

echo run_server started>>"%LOG%"
echo PY=%PY%>>"%LOG%"
echo APP_DIR=%APP_DIR%>>"%LOG%"

"%PY%" -u app.py 1>>"%LOG%" 2>>"%ERR%"
set "RC=%errorlevel%"
echo server exit code=%RC%>>"%ERR%"
exit /b %RC%
