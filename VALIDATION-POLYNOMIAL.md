# EXP-005 validation (local, 2026-10-18)

- Ran `PYTHONPATH=src python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --quick --seeds 1`; completed, printed synthetic diagnostics, no archived changes.
- Ran `PYTHONPATH=src python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --publish --seeds 8`; saved immutable 8-seed × 2-DGP analysis, CSV/JSON/NPZ, PNG and SVG under EXP-005 runs and linked actual figures in canonical note.
- Ran `PYTHONPATH=src python -m pytest -q`: **39 passed** (original tests + 6 new mathematical tests).
- Ran `python scripts/validate_research.py`: validated **5 numbered experiments**, matching run folders, image embeds and per-file SHA-256.
- Ran `PYTHONPATH=src python -m compileall -q src tests market_geometry/code/005-orthogonal_polynomial_geometry`: passed.
- `ruff` and `sphinx` are unavailable in this Python runtime; do not claim their validation. Both are specified in the project's Conda environment and CI protocol.
- All input asset returns are synthetic. Experiment AUROC is retrospective and its reported uncertainty is across seeded panels, not independent rolling windows. No real-stock inference.
- Two experiment caveats: signed lag-1 AUROC below 0.5 in the AR(2) test is score-direction-specific; the normalized-time Legendre derivative weight is not directly scale-matched to EXP-001 daily-unit spline derivative weight.
- Source inspiration and prior art are linked in `market_geometry/notes/005-orthogonal_polynomial_geometry/README.md`.
