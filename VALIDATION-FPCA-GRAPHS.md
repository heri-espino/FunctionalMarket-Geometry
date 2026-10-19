# EXP-006 validation ledger — 2026-10-18 local time

- Executed **8 independent simulation seeds**, AR(1) switch, AR(2) switch,
  and unshifted-parameter AR(1) null.
- Stored per-seed CSV, aggregate CSV, exemplar NPZ, JSON, 6 genuine PNG and
  6 SVG figures, SHA-256 manifest, and figure embeds in canonical note.
- Exact experiment run ID and aggregate statistics: see EXP-006 note/run.
- Regression tests cover exact metric recovery from Hilbert coordinates,
  FPCA full-distance preservation/truncation, symmetric weighted kNN,
  normalized Laplacian, fixed calibration bandwidth, and no forward leakage.
- `python -m pytest -q`: 47 passed in local environment.
- `python scripts/validate_research.py`: verified 6 experiment groups and
  all existing immutable run SHA-256 hashes.
- Ruff and Sphinx not available as installed modules in this container;
  CI/environment.yml requires them. Do not mark those checks as passed.
- No edits or pushes to a GitHub remote and no real market data.
