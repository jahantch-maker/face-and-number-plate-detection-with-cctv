@echo off
title Gate Vision - update
cd /d %~dp0
echo.
if exist .venv\Scripts\python.exe (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
%PY% update.py %*
set "RC=%errorlevel%"
if "%RC%"=="2" (
  echo.
  echo  Nothing to do. Gate Vision keeps running as it is.
  pause
  exit /b 0
)
if not "%RC%"=="0" (
  echo.
  echo  The update did not finish - see the message above.
  echo  If Gate Vision is stopped, run start_all.bat to start it again.
  pause
  exit /b 1
)
echo.
echo  Starting Gate Vision with the new version...
call start_all.bat
exit /b 0
