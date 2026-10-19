# EXP-003 validation record — 2026-10-18

- Original root experiments 001 and 002 preserved without overwriting their archived evidence.
- 2007 Guerrero publisher abstract and 2017 follow-on article checked for attribution of the controlled S method. Gaussian ML derived from the variance-ratio model; **not** claimed as an original Guerrero estimator.
- `PYTHONPATH=src python -m pytest -q`: **PASS** with 21 tests (after new module and 5 new tests), including direct masked-refit checks for d=1..4 and exact outer-index invariance of both LOOCV and Gaussian-ML selection.
- `python scripts/validate_research.py`: **PASS** for all three matching numbered experiment families and immutable archive SHA-256 manifests.
- `PYTHONPATH=src python -m compileall -q src/market_geometry`: **PASS**.
- EXP-003 `--quick`: **PASS** (`--n 26 --seeds 1`), no archived output created.
- EXP-003 `--publish --seeds 4 --n 48`: **PASS**; immutable results, four PNG/SVG pairs, exact held-out predictions and run-level SHA-256 manifest. The original exploratory run was generated before a small robustness improvement to ML's numerical local-minimum bracket search. **A second immutable run** was generated afterwards and matches the current implementation: `run_20261018T221759917877Z_s20261020_n4`. Both archives are retained, labeled and validated; use the **second** run for reporting.
- Existing EXP-001 and EXP-002 `--quick` smoke tests: **PASS** (one seed for EXP-002).
- `ruff`: **NOT AVAILABLE** (no installed Python module).
- `sphinx`: **NOT AVAILABLE** (no installed Python module).
- Synthetic only; no real financial data, causal inference, or manuscript-ready claims.
