@echo off
cd /d "%~dp0.."
.venv\Scripts\python training\evaluate_ocr.py %*
echo.
pause
