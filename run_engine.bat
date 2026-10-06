@echo off
cd /d %~dp0
:loop
.venv\Scripts\python run_engine.py >> engine.log 2>&1
echo engine stopped, restarting in 10 s >> engine.log
timeout /t 10 /nobreak >nul
goto loop
