@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

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
set "URL=http://127.0.0.1:7860"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

echo.
echo ================================================
echo   FaceHighlight - Click to Start
echo ================================================
echo.

if not exist "%PY%" (
    echo [FAIL] Missing bundled python:
    echo   %PY%
    goto :fail
)
if not exist "%APP_DIR%\app.py" (
    echo [FAIL] Missing app.py:
    echo   %APP_DIR%\app.py
    goto :fail
)
if not exist "%APP_DIR%\highlight_removal" (
    echo [FAIL] Missing module folder:
    echo   %APP_DIR%\highlight_removal
    goto :fail
)
if not exist "%APP_DIR%\models\face_landmarker.task" (
    echo [FAIL] Missing face model:
    echo   %APP_DIR%\models\face_landmarker.task
    goto :fail
)
if not exist "%ROOT%data" (
    echo [FAIL] Missing data folder:
    echo   %ROOT%data
    goto :fail
)

del "%LOG%" 2>nul
del "%ERR%" 2>nul

call :kill_port
timeout /t 1 /nobreak >nul

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -Uri '%URL%' -UseBasicParsing -TimeoutSec 2).StatusCode | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo Service already running. Opening browser...
    start "" "%URL%"
    pause
    exit /b 0
)

echo [1/4] Checking Python imports...
set "PYTHONPATH=%APP_DIR%"
cd /d "%APP_DIR%"
"%PY%" -c "import highlight_removal; import gradio, cv2, mediapipe; print('imports ok')" 1>>"%LOG%" 2>>"%ERR%"
if %errorlevel% neq 0 (
    echo [FAIL] Import check failed.
    goto :show_error
)

echo [2/4] Starting server process...
start "FaceHighlight-Server" /MIN "%ROOT%run_server.bat"

echo [3/4] Waiting for web UI (up to 180 seconds)...
set /a WAIT=0
:wait_loop
if %WAIT% geq 90 goto :wait_timeout

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -Uri '%URL%' -UseBasicParsing -TimeoutSec 2).StatusCode | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 goto :start_ok

if exist "%ERR%" (
    for /f "usebackq delims=" %%L in ("%ERR%") do (
        echo %%L | findstr /i "Traceback OSError ModuleNotFound \"Cannot find empty port\"" >nul && goto :server_crashed
    )
)

timeout /t 2 /nobreak >nul
set /a WAIT+=1
goto :wait_loop

:server_crashed
echo [FAIL] Server crashed during startup.
goto :show_error

:wait_timeout
echo [4/4] Timeout - server did not open %URL%
goto :show_error

:start_ok
echo [4/4] Server is ready.
start "" "%URL%"
echo.
echo ================================================
echo   START OK - Browser opened
echo   URL: %URL%
echo ================================================
echo.
echo To stop: run stop.bat
echo.
pause
exit /b 0

:show_error
echo.
if exist "%ERR%" (
    echo ========== ERROR LOG ==========
    type "%ERR%"
    echo ===============================
) else (
    echo No startup_error.log written.
)
if exist "%LOG%" (
    echo.
    echo ========== STARTUP LOG ==========
    type "%LOG%"
    echo ================================
)
goto :fail

:fail
echo.
echo Startup failed.
echo Please send startup_error.log and startup.log for support.
echo.
pause
exit /b 1

:kill_port
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$pids = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach($p in $pids){ Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }"
exit /b 0
