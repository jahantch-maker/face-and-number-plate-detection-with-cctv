@echo off
cd /d "%~dp0.."
.venv\Scripts\python training\collect_plates.py %*
echo.
pause
