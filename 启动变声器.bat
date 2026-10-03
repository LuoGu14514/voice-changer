@echo off
rem Launch the voice changer GUI (no console window).
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto :novenv
start "" ".venv\Scripts\pythonw.exe" voice_changer.py
exit /b 0

:novenv
echo [!] .venv not found. Create it with:
echo     python -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
pause
exit /b 1
