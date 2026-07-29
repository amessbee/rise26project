@echo off
setlocal
cd /d "%~dp0"
set PORT=8081

echo.
echo U.S. INFRASTRUCTURE STRESS MONITOR
echo ==================================
echo Starting at http://127.0.0.1:8081
echo.
echo Keep this window open while using the GUI.
echo Press Ctrl+C to stop the server.
echo.

where py >nul 2>nul
if not errorlevel 1 (
    py server.py
    goto :end
)

where python >nul 2>nul
if not errorlevel 1 (
    python server.py
    goto :end
)

echo Python 3 was not found.
echo Install Python 3 and run this file again.
pause

:end
endlocal
