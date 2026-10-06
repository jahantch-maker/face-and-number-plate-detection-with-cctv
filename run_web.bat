@echo off
cd /d %~dp0
:loop
.venv\Scripts\python run_web.py >> web.log 2>&1
timeout /t 10 /nobreak >nul
goto loop
