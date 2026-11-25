# Instalação local no Windows (PowerShell), sem Docker.
# Se der erro de permissão: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

python -m scripts.download_movielens
python -m scripts.seed

Write-Host "`nPronto! Para subir a API:  .\scripts\executar.ps1"
