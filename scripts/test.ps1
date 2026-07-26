# Run the offline test suite with coverage.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

uv run pytest --cov=src/hcmaic --cov-report=term
exit $LASTEXITCODE
