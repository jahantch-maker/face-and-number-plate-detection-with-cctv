@echo off
cd /d "%~dp0.."
.venv\Scripts\python training\pack_plates.py
echo.
pause
