@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

echo Stopping FaceHighlight on port 7860 ...
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$pids = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; if(-not $pids){ Write-Host 'No running service.'; exit 0 }; foreach($p in $pids){ try { Stop-Process -Id $p -Force -ErrorAction Stop; Write-Host ('Stopped PID ' + $p) } catch { Write-Host ('Failed PID ' + $p) } }"
echo Done.
pause
