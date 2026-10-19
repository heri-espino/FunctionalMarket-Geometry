# Local validation — 2026-10-18

**Scope:** Local ZIP-build session, *not* GitHub CI or an independently reproduced scientific result.

| Check | Observed status |
|---|---|
| Synthetic full-run numerical+figure generator | **Completed:** archived under `market_geometry/results/001-temporal_regime_pilot/runs/` with matching PNG/SVG, raw NPZ, CSV, JSON and hash manifest. |
| Unit tests | **Passed:** 7 tests. |
| Research directory, image links and immutable artifact SHA-256 checks | **Passed:** `python scripts/validate_research.py`. |
| Python source compilation | **Passed:** `python -m compileall -q src scripts market_geometry/code`. |
| Quick CLI smoke experiment | **Passed:** console output; no tracked run created. |
| Ruff | **Not executed:** unavailable in container; package installation failed because outbound package-index access was unavailable. Configured in `pyproject.toml` and GitHub Actions. |
| Sphinx build with `-W` | **Not executed:** unavailable in container; configured in CI and user environment. |
| GitHub Actions | **Not executed:** repository has not been created/pushed. |
| Independent repeat seeds, real financial data, spectral/lagged baselines | **Not executed / planned only.** |

**Synthetic result caution:** one fixed random seed and manually selected AR regime design; post-label AUROC is descriptive and uses overlapping windows. An AUROC below 0.5 for Pearson is a **score-direction result**, not proof that no information is present in the method. The prototype should not be presented as competitive research evidence until lagged/spectral baselines, replicated negative controls and out-of-time evaluation exist.
