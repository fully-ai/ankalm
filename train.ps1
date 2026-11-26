Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { & .\install.ps1 }
& ".venv\Scripts\python.exe" -m ankalm.cli train --config configs/small.yaml @args
