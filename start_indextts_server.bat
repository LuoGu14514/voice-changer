@echo off
REM Start the IndexTTS-2.5 HTTP service on port 9881.
REM Must run from D:\vc (an ASCII junction of D:\代码\voice_changer): the
REM kaldifst-based wetext FST loader in IndexTTS cannot open paths that
REM contain non-ASCII characters.
setlocal
cd /d D:\vc
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if "%1"=="" (set PORT=9881) else (set PORT=%1)
echo [start_indextts_server] port=%PORT%
REM Relative venv path on purpose: we are already in D:\vc.
".\_indextts_venv\Scripts\python.exe" -u indextts_server.py --port %PORT%
