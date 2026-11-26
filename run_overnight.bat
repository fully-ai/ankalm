@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
if not exist ".venv\Scripts\python.exe" call install.bat
echo [AnkaLM] Kalici Gece Boyu (Overnight) Egitim Denetleyicisi baslatiliyor...
".venv\Scripts\python.exe" training_supervisor.py --detach %*
".venv\Scripts\python.exe" training_status.py
endlocal
