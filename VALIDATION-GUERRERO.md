# Reproducibility validation — EXP-002

Date: 2026-10-18 (UTC).

- GitHub **Trend-Estimation** source read through the connected repository: README, original Guerrero notes, theoretical contributions, CV method definition.
- Original **FunctionalMarket-Geometry** project restored from its previously created ZIP into an independent working copy; EXP-001 files and archived historical figures were preserved unchanged.
- Nine explicit model comparators in EXP-002, including signed lag-one cross-correlation.
- `python market_geometry/code/002-guerrero_smoothing/run.py --quick --seeds 1`: **PASSED**.
- `PYTHONPATH=src pytest -q`: **PASSED**, 16 tests total, including nine new PLS mathematics/chronology tests.
- `python scripts/validate_research.py`: **PASSED**, both numbered experiment families and stored SHA-256 manifests.
- `python market_geometry/code/002-guerrero_smoothing/run.py --publish --seeds 12`: **PASSED**; genuine new archive, numerical CSV/JSON, PNG/SVG figures.
- `python -m compileall -q src/market_geometry/guerrero.py`: **PASSED**.
- `ruff`: **NOT AVAILABLE** in the execution environment; attempted install, offline dependency index failed. It remains part of continuous integration.
- `sphinx`: **NOT AVAILABLE** in the execution environment; attempted install, offline dependency index failed. It remains part of continuous integration.
- No real stock returns were processed. No remote GitHub writes performed.

These checks do not prove financial relevance or publication novelty. The simulation selects a known AR-memory change; seed-level 1.000 AUROC for both lag-one cross-correlation and other methods demonstrates the baseline is strong, not functional superiority.
