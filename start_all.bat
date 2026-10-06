@echo off
cd /d %~dp0
if not exist config.yaml ( echo  Run install.bat first. & pause & exit /b 1 )
start "Gate Vision - Web" /min cmd /c run_web.bat
start "Gate Vision - Engine" /min cmd /c run_engine.bat
echo  Gate Vision is starting. Open http://localhost:8080 in your browser.
echo  Logs: engine.log and web.log in this folder.
timeout /t 8 >nul
start http://localhost:8080
