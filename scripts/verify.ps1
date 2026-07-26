# Full verification loop: build/import, types, lint, tests+coverage, fixture E2E.
# Mirrors VERIFICATION_REPORT.md. Fails fast on the first broken gate.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Gate($name, $script) {
    Write-Host "== $name ==" -ForegroundColor Cyan
    & $script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GATE FAILED: $name" -ForegroundColor Red
        exit 1
    }
}

Gate "Environment / import / CLI help" { uv run hcmaic --help | Out-Null; uv run python -c "import hcmaic" }
Gate "Type check (mypy)"              { uv run mypy src }
Gate "Lint (ruff)"                    { uv run ruff check src tests scripts }
Gate "Tests + coverage"               { uv run pytest --cov=src/hcmaic --cov-report=term }
Gate "Fixture E2E: validate"          { uv run hcmaic validate-data --input data/sample }
Gate "Fixture E2E: build-index"       { uv run hcmaic build-index --input data/sample --output artifacts/verify }
Gate "Fixture E2E: CLI search"        { uv run hcmaic search --index artifacts/verify --query "a solid red keyframe" --top-k 3 }
Gate "Fixture E2E: evaluate"          {
    uv run hcmaic evaluate --index artifacts/verify `
        --queries data/sample/queries.jsonl --qrels data/sample/qrels.jsonl `
        --out artifacts/verify/evaluation
}

Write-Host "`nAll verification gates passed." -ForegroundColor Green
