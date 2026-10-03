@echo off
rem Launch with a console so errors are visible (debug).
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" voice_changer.py %*
echo.
echo [exit code %ERRORLEVEL%]
pause
