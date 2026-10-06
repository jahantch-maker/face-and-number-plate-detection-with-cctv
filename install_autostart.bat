@echo off
cd /d %~dp0
net session >nul 2>nul
if errorlevel 1 (
  echo  Right-click this file and choose "Run as administrator".
  pause
  exit /b 1
)
schtasks /create /tn "GateVision Engine" /tr "\"%~dp0run_engine.bat\"" /sc onstart /ru SYSTEM /rl HIGHEST /f
schtasks /create /tn "GateVision Web" /tr "\"%~dp0run_web.bat\"" /sc onstart /ru SYSTEM /rl HIGHEST /f
echo.
echo  Done. Gate Vision will start by itself whenever this PC starts.
echo  (Also set the PC BIOS to "power on after power failure".)
pause
