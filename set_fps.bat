@echo off
cd /d "%~dp0"
.venv\Scripts\python set_fps.py
echo.
set /p N=Type the new number (15 is a good choice, 0 = fastest) and press Enter, or just press Enter to keep it: 
if "%N%"=="" goto end
.venv\Scripts\python set_fps.py %N%
:end
echo.
echo Now close the Gate Vision windows and run start_all.bat again.
pause
