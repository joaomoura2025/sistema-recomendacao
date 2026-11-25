# Sobe a API localmente no Windows (PowerShell). Rode .\scripts\instalar.ps1 antes.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

.\.venv\Scripts\Activate.ps1
Write-Host "Swagger: http://localhost:8000/docs"
uvicorn app.main:app --reload --port 8000
