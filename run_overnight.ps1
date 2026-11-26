Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { & .\install.ps1 }
Write-Host "[AnkaLM] Kalici Gece Boyu (Overnight) Egitim Denetleyicisi baslatiliyor..." -ForegroundColor Cyan
& ".venv\Scripts\python.exe" training_supervisor.py --detach @args
Start-Sleep -Seconds 1
& ".venv\Scripts\python.exe" training_status.py
