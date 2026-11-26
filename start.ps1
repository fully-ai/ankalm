Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { & .\install.ps1 }
& ".venv\Scripts\python.exe" -m ankalm.cli serve --host 127.0.0.1 --port 8420 --open-browser
