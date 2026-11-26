@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
if not exist ".venv\Scripts\python.exe" call install.bat
echo [AnkaLM] Egitim en son kaydedilen kontrol noktasindan (latest) devam ettiriliyor...
".venv\Scripts\python.exe" -m ankalm.cli train --config configs/small.yaml --resume latest %*
endlocal
