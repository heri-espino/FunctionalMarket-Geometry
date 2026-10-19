# 2026-10-18 — Expanding to competing functional trend estimators

## Why

User asked to compare B-splines and d-th difference trends with potentially better trend estimators. Wanted same ProximityGraphs-Isomap-style structure and continued ZIP project. Prior EXP-003 only compared d=1–4 PLS selectors.

## What was done

Added `src/market_geometry/method_comparison.py` implementing PLS d=1..4, linear/cubic regression B-splines, P-splines with coefficient penalty orders 1–3, exact natural cubic smoothing splines, RBF kernel ridge and polynomial benchmarks. Built a 49-candidate bank. Each method's tuning takes place INSIDE nested leave-one-out exclusion. The All-families selector competes across family and hyperparameters INSIDE that same inner selection, to avoid using outer results as training labels. Tested direct deletion against independently refitted B-splines, P-splines, RBF kernel ridge, and natural cubic smoothing splines. Existing archived experiments left intact.

## Evidence actually produced

New `market_geometry/code/004-cross_family_methods/run.py` produces CSV scores, JSON, SHA-256 manifest and PNG/SVG figures in immutable run folders. A 6-seed pilot and a 30-seed followup completed and remain separately archived. 30-seed mean observed outer-LOOCV RMSE: PLS d1 ≈0.2500, RBF ≈0.2545, natural cubic ≈0.2549. These are synthetic reconstruction metrics, NOT real-market forecasts or regime-network scores.

## Math/limitations and next question

All shortlisted models are fixed-grid symmetric linear smoothers with exact pair-deletion identities. Guerrero's μ plug-in, LOESS, L1 trend filtering, Kalman likelihood fitting and wavelets are not comparable via this shortcut and remain future work. Compare reconstruction, oracle latent recovery and chronological forecasting separately. Need 30+ independent new scenarios for reproducible research and rigorous claims; finance-specific out-of-time and network metrics are currently absent.
