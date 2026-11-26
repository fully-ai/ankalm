@echo off
setlocal
cd /d "%~dp0"
echo [AnkaLM] Yerel Python 3.12 ortami ve bagimliliklar kuruluyor...

set PYTHON_CMD=python
if exist "C:\Users\ubeyt\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD="C:\Users\ubeyt\AppData\Local\Programs\Python\Python312\python.exe"
) else (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 set PYTHON_CMD=py -3.12
)

if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv --system-site-packages
)

set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" -m pip install -e . --no-build-isolation --no-deps >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements\base.txt
)

echo [AnkaLM] Donanim kesfi calistiriliyor...
".venv\Scripts\python.exe" -m ankalm.cli doctor
echo [AnkaLM] Kurulum basariyla tamamlandi!
endlocal
