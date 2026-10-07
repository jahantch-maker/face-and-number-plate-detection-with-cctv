@echo off
cd /d "%~dp0"
.venv\Scripts\python set_channel.py
echo.
set /p CAM=Type the camera name to change (for example in_plate), or press Enter to leave: 
if "%CAM%"=="" goto end
set /p CH=Type the correct NVR channel number (for example 8): 
.venv\Scripts\python set_channel.py %CAM% %CH%
:end
echo.
pause
