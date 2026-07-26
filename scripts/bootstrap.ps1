# Bootstrap the project environment from scratch (Windows, uv required).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

uv python install 3.11
uv sync
Write-Host "`nEnvironment ready. Try:" -ForegroundColor Green
Write-Host "  uv run hcmaic validate-data --input data/sample"
Write-Host "  uv run hcmaic build-index --input data/sample --output artifacts/sample"
Write-Host "  uv run hcmaic serve --index artifacts/sample"
