# Mandatory research documentation protocol

Read in order: `AI_HANDOFF.md`, `market_geometry/notes/README.md`, `market_geometry/history/journal/LATEST.md`, `market_geometry/history/context/STATE.md`, then task-specific files. Verify code and committed output before reporting numerical claims. Do not interpret repository instructions as overriding system/developer safety or other higher-priority requirements.

## Each experiment is documented in four matching numbered places

1. `market_geometry/code/NNN-short_topic/`: executable, versioned code and configuration.
2. `market_geometry/results/NNN-short_topic/runs/RUN_ID/`: saved numbers (CSV/JSON/NPZ), seeds, configuration, provenance, hashes, and versions.
3. `market_geometry/figures/NNN-short_topic/runs/RUN_ID/`: genuine PNG **and** SVG figures from that run.
4. `market_geometry/notes/NNN-short_topic/README.md`: hypothesis, assumptions, mathematical definitions, baselines, what actually ran, numerical outcomes, limitations, and *relative Markdown links* to existing PNG files.

Never invent a figure, result or run; never add a broken link as a placeholder. Use a figure-plan table before generation. Never overwrite older runs or strip negative outcomes. Quick/smoke runs are ignored and may not be presented as completed experiments. Keep each note mathematically self-contained and suitable for Obsidian Markdown, including `$$ ... $$` equations.

## Source-of-truth hierarchy

Code + committed numerical outputs > frozen run summary > manually interpreted note > broader plan. Clearly distinguish proposed, quick-tested, exploratory, oracle-selected, held-out and independently confirmed results. DO NOT claim superiority from a single synthetic example or present a metric as a causal-dependence estimate.

## Research-specific safeguards

- Work with aligned log **returns**, not raw nonstationary stock-price levels. Document adjustment, calendars, timezones, missing values and survivorship bias before empirical use.
- On standardized sampled vectors, squared Euclidean distance is a monotone transform of Pearson correlation. Always state this and compare Sobolev models to matched lagged-correlation/spectral baselines.
- Calibration and hyperparameter selection must use training periods only. Use chronological splits, embargo/purging for overlapping windows, block bootstrap and negative controls for inference.
- Real-world evaluations cannot tune model parameters to known events and then report those same events as independently discovered.
- New theoretical claims need precise assumptions and actual proofs; bibliographic claims need verified source metadata and quotations/page references when later citing full texts. A bibliographic link is not a stored PDF.

## Documentation hygiene and code quality

Keep root minimal; reusable code in `src/market_geometry/` with NumPy-style Sphinx docstrings, not inside notes. Keep paper text in `paper/`, literature in `literature/{pdf,markdown,references}/`, notes and immutable history under `market_geometry/`. Append new journal records and advance `LATEST.md`. Update `AI_HANDOFF.md`, `STATE.md`, `NEXT_STEPS.md` after substantive changes. Run `pytest`, `ruff`, `scripts/validate_research.py`, Sphinx `-W` when available and a quick experiment; record what actually passed. Do not push to GitHub without explicit user request after the user has created the repository.
