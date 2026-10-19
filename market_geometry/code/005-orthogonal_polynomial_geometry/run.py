"""EXP-005: Polynomial coordinates of smoothed return paths and Hilbert metrics.

Run from the repository root:
  python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --quick
  python market_geometry/code/005-orthogonal_polynomial_geometry/run.py --publish --seeds 8

Only generated synthetic returns are used. All models and metric weights are
fixed before seeing regime labels; no market outperformance is claimed.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from market_geometry.guerrero import fit_panel, lag_one_crosscorr  # noqa: E402
from market_geometry.metrics import (  # noqa: E402
    SplineSettings, correlation_distance, spline_metric, standardize_windows,
)
from market_geometry.polynomial_geometry import (  # noqa: E402
    evaluate_legendre, pairwise_polynomial_distance, project_onto_legendre,
)
from market_geometry.rolling import reference_deviation  # noqa: E402
from market_geometry.simulation import SimulationSettings, simulate_market  # noqa: E402

EXPERIMENT = "005-orthogonal_polynomial_geometry"
NOTE = ROOT / "market_geometry" / "notes" / EXPERIMENT / "README.md"


def lagged_symmetric_feature(z: np.ndarray, lag: int) -> np.ndarray:
    """Symmetrized signed cross-asset lag correlation, not a metric."""
    x, y = z[:, lag:], z[:, :-lag]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.maximum(1e-12, np.linalg.norm(x, axis=1)[:, None] * np.linalg.norm(y, axis=1)[None, :])
    corr = (x @ y.T) / denom
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 0.0)
    return np.clip(corr, -1.0, 1.0)


def simulate_lag2(seed: int, *, n_assets: int, n_steps: int, boundary: int,
                  n_groups: int = 3, strength: float = 0.7,
                  phi_before: float = 0.75, phi_after: float = -0.65):
    """Group factor f[t]=phi*f[t-2]+white innovations, stationary var one.

    Population contemporaneous factor covariance is unchanged and lag-one
    autocorrelation remains zero in a stationary regime, but lag two changes.
    Transients near the regime change are excluded from scoring.
    """
    if n_assets % n_groups or not 0 < strength < 1 or boundary < 20:
        raise ValueError("Invalid simulation design")
    rng = np.random.default_rng(seed)
    factors = np.zeros((n_groups, n_steps))
    for start, stop, phi in [(0, boundary, phi_before), (boundary, n_steps, phi_after)]:
        factors[:, start:start + 2] = rng.normal(size=(n_groups, 2))
        for t in range(start + 2, stop):
            factors[:, t] = phi * factors[:, t - 2] + np.sqrt(1 - phi ** 2) * rng.normal(size=n_groups)
    groups = np.repeat(np.arange(n_groups), n_assets // n_groups)
    returns = np.sqrt(strength) * factors[groups] + np.sqrt(1 - strength) * rng.normal(size=(n_assets, n_steps))
    return returns, groups


def compute_matrices(window: np.ndarray) -> tuple[dict, dict]:
    """Calculate predefined method representations, same window and assets.

    Different smoothing settings are not LOOCV-optimized in this experiment.
    The point is a first geometry benchmark, not fitted-return RMSE ranking.
    """
    z = standardize_windows(window)
    matrices = {
        "pearson": correlation_distance(z),
        "lag1": lag_one_crosscorr(z),
        "lag2": lagged_symmetric_feature(z, 2),
        "bspline_h1": spline_metric(
            z, SplineSettings(n_basis=18, degree=3, ridge=0.02, derivative_weight=0.75)
        ),
    }
    representation = {"raw": z}
    for d in (1, 2):
        representation[f"pls{d}"] = fit_panel(z, 0.50, d=d, mean_mode="zero")
    representation["pls2_muG"] = fit_panel(z, 0.50, d=2, mean_mode="guerrero")
    coefficients = {}
    for label, trends in representation.items():
        for degree in (3, 7):
            coef = project_onto_legendre(trends, degree)
            coefficients[(label, degree)] = coef
            for metric, alpha in (("l2", 0.0), ("h1", 0.01)):
                key = f"legendre_{label}_p{degree}_{metric}"
                matrices[key] = pairwise_polynomial_distance(coef, alpha)
    return matrices, {"z": z, "trends": representation, "coeff": coefficients}


def rolling_panel(returns: np.ndarray, *, window: int, stride: int):
    """Evaluate only trailing windows at their exclusive end indices."""
    ends = np.arange(window, returns.shape[1] + 1, stride)
    store = {}
    diagnostics = None
    for j, end in enumerate(ends):
        matrices, info = compute_matrices(returns[:, end - window:end])
        if j == 0:
            store = {name: [] for name in matrices}
            diagnostics = info
        for name, values in matrices.items():
            store[name].append(values)
    return ends, {name: np.stack(items) for name, items in store.items()}, diagnostics


def score_regime(matrices: dict, ends: np.ndarray, *, boundary: int, window: int):
    """Training/calibration-only network reference, with disjoint evaluation.

    Known simulation labels are used only to evaluate discrimination. Windows
    crossing the switch and initial calibration windows are excluded.
    """
    cal = ends <= boundary // 2
    pre = (ends > boundary // 2) & (ends <= boundary)
    post = ends >= boundary + window
    ev = pre | post
    if min(cal.sum(), pre.sum(), post.sum()) < 2:
        raise ValueError("Require >=2 calibration/pre/post windows")
    label = post[ev].astype(int)
    rows, time_scores = {}, {}
    for name, values in matrices.items():
        score, center = reference_deviation(values, cal)
        rows[name] = {
            "auroc": float(roc_auc_score(label, score[ev])),
            "calibrated_pre_mean": float(score[pre].mean()),
            "calibrated_post_mean": float(score[post].mean()),
        }
        time_scores[name] = score
    return rows, time_scores, (cal, pre, post)


def run(*, publish: bool = False, quick: bool = False, seeds: int = 8) -> dict:
    if publish and quick:
        raise ValueError("Quick mode must not publish exploratory evidence")
    if seeds < 1:
        raise ValueError("At least one seed required")
    n_steps, boundary, window, stride = (288, 144, 48, 12) if quick else (480, 240, 64, 12)
    result_rows, exemplar = [], None
    for seed_index in range(seeds):
        seed = 20261021 + seed_index
        for generator in ("ar1_memory", "ar2_memory"):
            if generator == "ar1_memory":
                panel, _, groups = simulate_market(SimulationSettings(
                    seed=seed, n_assets=12, n_groups=3, n_steps=n_steps, boundary=boundary,
                    group_strength=0.7, phi_before=0.75, phi_after=-0.65,
                ))
            else:
                panel, groups = simulate_lag2(seed, n_assets=12, n_steps=n_steps, boundary=boundary)
            ends, matrices, info = rolling_panel(panel, window=window, stride=stride)
            scores, t_scores, masks = score_regime(matrices, ends, boundary=boundary, window=window)
            for method, values in scores.items():
                result_rows.append({
                    "seed": seed, "dgp": generator, "method": method,
                    "auroc": values["auroc"],
                    "pre_score": values["calibrated_pre_mean"],
                    "post_score": values["calibrated_post_mean"],
                })
            if exemplar is None and generator == "ar2_memory":
                exemplar = (panel, groups, ends, info, matrices, t_scores, masks)
    names = sorted({r["method"] for r in result_rows})
    aggregated = []
    for dgp in ("ar1_memory", "ar2_memory"):
        for method in names:
            v = np.asarray([r["auroc"] for r in result_rows if r["dgp"] == dgp and r["method"] == method])
            aggregated.append({"dgp": dgp, "method": method, "mean_auroc": float(v.mean()),
                               "sd_auroc": float(v.std(ddof=1)) if len(v) > 1 else None,
                               "n_independent_seeds": len(v)})
    summary = {
        "status": "exploratory synthetic representation and regime discrimination study",
        "experiment": EXPERIMENT,
        "n_seeds": seeds,
        "design": {
            "n_assets": 12, "n_steps": n_steps, "boundary": boundary, "window": window,
            "stride": stride, "difference_order": [1, 2], "s_fixed": 0.5,
            "legendre_degrees": [3, 7], "h1_normalized_time_weight": 0.01,
            "comparison": "train-reference network deviation AUROC; not LOOCV/forecast RMSE",
            "selection": "predeclared representations, no regime-label-based tuning",
        },
        "aggregate": aggregated,
        "limits": [
            "Synthetic labels known only for retrospective scoring; no observed equity market panels.",
            "The two regimes are intentionally constructed; a lag-two baseline is essential.",
            "Overlapping windows are not independent; variability is described across seeds.",
            "Choosing the best AUROC from this method grid would be post-hoc model selection.",
            "Polynomial projection error is not held-out trend reconstruction or out-of-time forecast RMSE.",
        ],
    }
    if not publish:
        print(json.dumps(summary, indent=2))
        return summary
    rid = "run_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + f"_n{seeds}"
    result_dir = ROOT / "market_geometry" / "results" / EXPERIMENT / "runs" / rid
    figure_dir = ROOT / "market_geometry" / "figures" / EXPERIMENT / "runs" / rid
    result_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)

    def write_csv(path: Path, rows: list[dict]):
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(result_dir / "seed_method_auroc.csv", result_rows)
    write_csv(result_dir / "mean_auroc_by_regime_and_method.csv", aggregated)
    panel, groups, ends, info, matrices, times, masks = exemplar
    np.savez_compressed(
        result_dir / "exemplar_synthetic_networks.npz",
        returns=panel, asset_groups_for_diagnostics=groups, ends=ends,
        raw_legendre3=info["coeff"][("raw", 3)],
        pls1_legendre3=info["coeff"][("pls1", 3)],
        lag2_networks=matrices["lag2"],
        polynomial_networks=matrices["legendre_pls1_p7_h1"],
    )

    # Main figure: per-family comparisons by DGP, not a selectively chosen winner.
    figure_methods = [
        "pearson", "lag1", "lag2", "bspline_h1",
        "legendre_raw_p3_l2", "legendre_raw_p3_h1", "legendre_raw_p7_h1",
        "legendre_pls1_p3_l2", "legendre_pls1_p3_h1", "legendre_pls1_p7_h1",
        "legendre_pls2_p3_h1", "legendre_pls2_p7_h1",
        "legendre_pls2_muG_p7_h1",
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(figure_methods))
    for idx, (dgp, text_label) in enumerate((
        ("ar1_memory", "AR(1) factor shift"), ("ar2_memory", "AR(2) factor shift"),
    )):
        val = [next(r["mean_auroc"] for r in aggregated if r["dgp"] == dgp and r["method"] == m) for m in figure_methods]
        ax.scatter(x + 0.12 * idx - 0.06, val, s=38, label=text_label, zorder=3)
        ax.plot(x + 0.12 * idx - 0.06, val, linewidth=0.75, alpha=0.65)
    ax.set_xticks(x, figure_methods, rotation=55, ha="right", fontsize=8)
    ax.set_ylim(-0.025, 1.04)
    ax.set_ylabel("Mean regime AUROC over independent simulated seeds")
    ax.set_title("EXP-005 | Polynomial market geometry vs essential lag baselines")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_dir / "regime_auroc_by_method.png", dpi=180)
    fig.savefig(figure_dir / "regime_auroc_by_method.svg")
    plt.close(fig)

    # Example curves: polynomial approximation AFTER noisy-path smoothing.
    u = np.linspace(0, 1, window)
    raw = info["z"][0]
    trend = info["trends"]["pls1"][0]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(u, raw, label="Raw standardized synthetic return", alpha=0.55, lw=1.0)
    ax.plot(u, trend, label="PLS-smoothed return path, d=1 / S=0.5", lw=2)
    for degree in (3, 7):
        curve = evaluate_legendre(info["coeff"][("pls1", degree)][0:1], u)[0]
        ax.plot(u, curve, label=f"Legendre degree {degree}", lw=1.6)
    ax.set(xlabel="Normalized within-window time, u", ylabel="Standardized return / fitted path",
           title="EXP-005 | Smooth first, then compress into a polynomial basis")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "polynomial_trend_gallery.png", dpi=180)
    fig.savefig(figure_dir / "polynomial_trend_gallery.svg")
    plt.close(fig)

    # Changing network detection scores over time, deliberately display lag2 baseline.
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for name in ("pearson", "lag1", "lag2", "legendre_raw_p7_h1", "legendre_pls1_p7_h1"):
        score = times[name]
        # Normalize using calibration-only variability, for visualization ONLY.
        cal = masks[0]
        denom = max(float(score[cal].std()), 1e-8)
        ax.plot(ends, (score - score[cal].mean()) / denom, label=name, linewidth=1.3)
    ax.axvline(boundary, linestyle="--", color="black", linewidth=1, label="True simulated change")
    ax.axvspan(boundary, boundary + window, color="gray", alpha=0.12)
    ax.set(xlabel="Exclusive trailing-window end index", ylabel="Standardized deviation score",
           title="EXP-005 | AR(2) memory change, illustrative single simulation")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "lag2_regime_timeseries.png", dpi=180)
    fig.savefig(figure_dir / "lag2_regime_timeseries.svg")
    plt.close(fig)

    summary["run_id"] = rid
    summary["run_utc"] = datetime.now(UTC).isoformat()
    summary["software"] = {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rel_files = sorted(list(result_dir.glob("*")) + list(figure_dir.glob("*")))
    sha = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in rel_files if p.name != "sha256_manifest.json"}
    (result_dir / "sha256_manifest.json").write_text(json.dumps(sha, indent=2) + "\n", encoding="utf-8")
    # Link only completed, versioned figures and numeric evidence.
    with NOTE.open("a", encoding="utf-8") as f:
        f.write(f"\n## Completed exploratory run `{rid}`\n\n")
        f.write(f"Synthetic AR(1) and AR(2) memory switches; {seeds} independently seeded panels per DGP; 12 assets; {n_steps} dates. No market data. All hyperparameters specified before reading the labels.\n\n")
        f.write("| Method | AR(1) AUROC | AR(2) AUROC |\n|---|---:|---:|\n")
        for m in figure_methods:
            x1 = next(r["mean_auroc"] for r in aggregated if r["dgp"] == "ar1_memory" and r["method"] == m)
            x2 = next(r["mean_auroc"] for r in aggregated if r["dgp"] == "ar2_memory" and r["method"] == m)
            f.write(f"| `{m}` | {x1:.3f} | {x2:.3f} |\n")
        f.write(f"\n[Full per-seed scores](../../results/{EXPERIMENT}/runs/{rid}/seed_method_auroc.csv) · [Summary and design](../../results/{EXPERIMENT}/runs/{rid}/summary.json) · [SHA256 evidence](../../results/{EXPERIMENT}/runs/{rid}/sha256_manifest.json)\n\n")
        for name in ("regime_auroc_by_method", "polynomial_trend_gallery", "lag2_regime_timeseries"):
            f.write(f"![EXP-005 {name}](../../figures/{EXPERIMENT}/runs/{rid}/{name}.png)\n\n")
        f.write("**Interpretation:** This benchmark tests whether polynomial compression preserves usable changes in network shape. These AUROCs are not forecasts, and a winning score is not proof of novelty. Lag-two correlation is a crucial comparator in the AR(2) model.\n")
    print(json.dumps({"run_id": rid, "aggregate": aggregated, "results": str(result_dir)}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Short unarchived smoke test")
    parser.add_argument("--publish", action="store_true", help="Archive immutable results and plots")
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    run(quick=args.quick, publish=args.publish, seeds=args.seeds)
