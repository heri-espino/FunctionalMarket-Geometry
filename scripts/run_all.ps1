param(
    [switch]$Quick,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try {
    python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'pytest failed' }
    python -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw 'ruff failed' }
    if ($Publish -and $Quick) { throw 'Choose -Quick OR -Publish' }
    if ($Publish) {
        python market_geometry/code/001-temporal_regime_pilot/run.py --publish
        if ($LASTEXITCODE -ne 0) { throw 'EXP-001 failed' }
        python market_geometry/code/002-guerrero_smoothing/run.py --publish --seeds 12
        if ($LASTEXITCODE -ne 0) { throw 'EXP-002 failed' }
        python market_geometry/code/003-order_lambda_loocv/run.py --publish --seeds 4 --n 48
        if ($LASTEXITCODE -ne 0) { throw 'EXP-003 failed' }
        python market_geometry/code/004-cross_family_methods/run.py --publish --seeds 30 --n 48
        if ($LASTEXITCODE -ne 0) { throw 'EXP-004 failed' }
        python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --publish --seeds 8
        if ($LASTEXITCODE -ne 0) { throw 'EXP-005 failed' }
        python market_geometry/code/006-functional_pca_graphs/run.py --publish --seeds 8
        python market_geometry/code/007-similarity_targets/run.py --publish --seeds 100
    } else {
        python market_geometry/code/001-temporal_regime_pilot/run.py --quick
        if ($LASTEXITCODE -ne 0) { throw 'EXP-001 failed' }
        python market_geometry/code/002-guerrero_smoothing/run.py --quick
        if ($LASTEXITCODE -ne 0) { throw 'EXP-002 failed' }
        python market_geometry/code/003-order_lambda_loocv/run.py --quick
        if ($LASTEXITCODE -ne 0) { throw 'EXP-003 failed' }
        python market_geometry/code/004-cross_family_methods/run.py --quick
        if ($LASTEXITCODE -ne 0) { throw 'EXP-004 failed' }
        python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --quick --seeds 1
        if ($LASTEXITCODE -ne 0) { throw 'EXP-005 failed' }
        python market_geometry/code/006-functional_pca_graphs/run.py --quick --seeds 1
        python market_geometry/code/007-similarity_targets/run.py --quick --seeds 1
    }
    if ($LASTEXITCODE -ne 0) { throw 'experiment failed' }
    python scripts/validate_research.py
    if ($LASTEXITCODE -ne 0) { throw 'research validation failed' }
    python -m sphinx -b html -W docs docs/_build
    if ($LASTEXITCODE -ne 0) { throw 'Sphinx failed' }
    Write-Host 'Validation complete. No files were pushed to GitHub.'
} finally {
    Pop-Location
}
