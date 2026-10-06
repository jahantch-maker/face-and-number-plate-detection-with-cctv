@echo off
title Gate Vision - install
cd /d %~dp0
echo.
echo  Gate Vision installer
echo  =====================
python --version >nul 2>nul
if errorlevel 1 (
  echo  Python is not installed. Trying to install it automatically ^(needs internet^)...
  winget --version >nul 2>nul
  if not errorlevel 1 (
    winget install -e --id Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
    echo.
    echo  Python installed. Restarting this installer so it can see Python...
    timeout /t 3 /nobreak >nul
    start "" cmd /c "%~f0"
    exit /b 0
  )
  echo  Automatic install is not available on this PC.
  echo  The download page will open. Install Python 3.12 and TICK "Add python.exe to PATH",
  echo  then double-click install.bat again.
  start https://www.python.org/downloads/windows/
  pause
  exit /b 1
)
if not exist .venv (
  echo  Creating private Python environment...
  python -m venv .venv
  if errorlevel 1 ( echo  Could not create the environment. & pause & exit /b 1 )
)
.venv\Scripts\python install_deps.py
if errorlevel 1 ( echo. & echo  Install failed - see the message above. & pause & exit /b 1 )
.venv\Scripts\python setup_wizard.py
echo.
net session >nul 2>nul
if not errorlevel 1 (
  echo  Setting up automatic start with Windows...
  call install_autostart.bat
) else (
  echo  TIP: right-click install.bat and choose "Run as administrator" to also make
  echo  Gate Vision start by itself whenever this PC starts.
)
echo.
echo  To start now: double-click start_all.bat
pause
