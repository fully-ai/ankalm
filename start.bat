@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
if not exist ".venv\Scripts\python.exe" call install.bat
echo [AnkaLM] Yerel Yapay Zeka Sohbet ve Egitim Kontrol Merkezi baslatiliyor (http://127.0.0.1:8420)...
".venv\Scripts\python.exe" -m ankalm.cli serve --host 127.0.0.1 --port 8420 --open-browser
endlocal
