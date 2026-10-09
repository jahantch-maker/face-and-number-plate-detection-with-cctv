@echo off
cd /d "%~dp0.."
echo If Windows asks whether to allow access, choose Allow (so the phone can open the page).
.venv\Scripts\python training\label_plates.py %*
pause
