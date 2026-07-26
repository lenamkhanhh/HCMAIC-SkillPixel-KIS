# Build the fixture index (if needed) and serve the API + operator UI.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path "artifacts/sample/index_manifest.json")) {
    uv run hcmaic build-index --input data/sample --output artifacts/sample
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
uv run hcmaic serve --index artifacts/sample --port 8000
