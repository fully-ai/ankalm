@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" training_status.py
endlocal
