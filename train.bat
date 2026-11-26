@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
if not exist ".venv\Scripts\python.exe" call install.bat
echo [AnkaLM] Sifirdan Turkce dil modeli egitimi baslatiliyor...
".venv\Scripts\python.exe" -m ankalm.cli train --config configs/small.yaml %*
endlocal
