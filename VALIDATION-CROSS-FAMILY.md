# EXP-004 execution audit (2026-10-18)

- `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q` — 33 tests passed at creation.
- `OPENBLAS_NUM_THREADS=1 python market_geometry/code/004-cross_family_methods/run.py --quick` — passed.
- Two immutable `--publish` runs: seeds=6 and seeds=30; archived 3 PNG + 3 SVG files each, seed-level scores, first-panel outer predictions, summary and SHA-256 manifest.
- `python scripts/validate_research.py` — four corresponding numbered experiment groups and all archived hashes verified.
- Direct refit and leave-one/two-out target-leakage tests included.
- Sphinx and ruff were NOT available in this container; remain required in Conda / CI.
- This is reconstruction on synthetic data, not financial forecasting or regime identification.
