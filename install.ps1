$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "[AnkaLM] Yerel Python 3.12 ortami hazirlaniyor..." -ForegroundColor Cyan

$pyCmd = "python"
if (Test-Path "C:\Users\ubeyt\AppData\Local\Programs\Python\Python312\python.exe") {
    $pyCmd = "C:\Users\ubeyt\AppData\Local\Programs\Python\Python312\python.exe"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pyCmd -m venv .venv --system-site-packages
}

$env:PYTHONPATH = $PSScriptRoot
& ".venv\Scripts\python.exe" -m pip install -e . --no-build-isolation --no-deps 2>$null
& ".venv\Scripts\python.exe" -m ankalm.cli doctor
Write-Host "[AnkaLM] Kurulum tamamlandi!" -ForegroundColor Green
