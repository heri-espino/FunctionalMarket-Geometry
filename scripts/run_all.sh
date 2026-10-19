#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python -m ruff check .
if [[ "${1:-}" == "--publish" ]]; then
  python market_geometry/code/001-temporal_regime_pilot/run.py --publish
  python market_geometry/code/002-guerrero_smoothing/run.py --publish --seeds 12
  python market_geometry/code/003-order_lambda_loocv/run.py --publish --seeds 4 --n 48
  python market_geometry/code/004-cross_family_methods/run.py --publish --seeds 30 --n 48
  python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --publish --seeds 8
  python market_geometry/code/006-functional_pca_graphs/run.py --publish --seeds 8
else
  python market_geometry/code/001-temporal_regime_pilot/run.py --quick
  python market_geometry/code/002-guerrero_smoothing/run.py --quick
  python market_geometry/code/003-order_lambda_loocv/run.py --quick
  python market_geometry/code/004-cross_family_methods/run.py --quick
  python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --quick --seeds 1
  python market_geometry/code/006-functional_pca_graphs/run.py --quick --seeds 1
fi
python scripts/validate_research.py
python -m sphinx -b html -W docs docs/_build
echo 'Validation complete. No files were pushed to GitHub.'
