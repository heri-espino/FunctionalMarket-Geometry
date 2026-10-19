"""EXP-006: window-local functional PCA, spectral graphs, and matched baselines.

From repository root:
  python market_geometry/code/006-functional_pca_graphs/run.py --quick --seeds 1
  python market_geometry/code/006-functional_pca_graphs/run.py --publish --seeds 8

This study is entirely synthetic; regime labels are used only after features
and calibration rules have been fixed. No profit or forecast claim is made.
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

from market_geometry.functional_pca_graph import (  # noqa: E402
    fit_functional_pca,
    hilbert_coordinates,
    pairwise_euclidean,
    reference_feature_score,
    rolling_graph_features,
)
from market_geometry.guerrero import fit_panel  # noqa: E402
from market_geometry.metrics import correlation_distance, standardize_windows  # noqa: E402
from market_geometry.polynomial_geometry import project_onto_legendre  # noqa: E402
from market_geometry.simulation import SimulationSettings, simulate_market  # noqa: E402

EXP = "006-functional_pca_graphs"
NOTE = ROOT / "market_geometry" / "notes" / EXP / "README.md"
LABELS = {
    "pearson": "Pearson correlation",
    "lag1": "Signed lag-1 correlation",
    "lag2": "Signed lag-2 correlation",
    "full_pls_h1": "Full PLS-H1 coefficients",
    "fpca_pls_k2": "FPCA PLS / 2 PCs",
    "fpca_pls_k3": "FPCA PLS / 3 PCs",
    "fpca_pls_k5": "FPCA PLS / 5 PCs",
    "fpca_raw_k3": "FPCA raw / 3 PCs",
}


def lagged_symmetric_correlation(z: np.ndarray, lag: int) -> np.ndarray:
    """Symmetric signed lagged cross-asset correlation (not a metric)."""
    a, b = z[:, lag:], z[:, :-lag]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    c = a @ b.T / np.maximum(1e-12, denom)
    c = np.clip(0.5 * (c + c.T), -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def window_features(window: np.ndarray):
    """Build fixed candidate features from window data only (no regime labels)."""
    z = standardize_windows(window)
    raw = hilbert_coordinates(project_onto_legendre(z, 7), alpha=0.01)
    trend = fit_panel(z, 0.50, d=1, mean_mode="zero")
    pls = hilbert_coordinates(project_onto_legendre(trend, 7), alpha=0.01)
    pf = fit_functional_pca(pls, components=5)
    rf = fit_functional_pca(raw, components=3)
    lag1 = lagged_symmetric_correlation(z, 1)
    lag2 = lagged_symmetric_correlation(z, 2)
    pearson_distance = correlation_distance(z)
    dists = {
        "pearson": pearson_distance,
        "lag1": np.sqrt(np.maximum(0.0, 2.0 * (1.0 - lag1))),
        "lag2": np.sqrt(np.maximum(0.0, 2.0 * (1.0 - lag2))),
        "full_pls_h1": pairwise_euclidean(pls),
        "fpca_pls_k2": pairwise_euclidean(pf.scores[:, :2]),
        "fpca_pls_k3": pairwise_euclidean(pf.scores[:, :3]),
        "fpca_pls_k5": pairwise_euclidean(pf.scores[:, :5]),
        "fpca_raw_k3": pairwise_euclidean(rf.scores),
    }
    direct = {
        "pearson_direct": 1.0 - (pearson_distance**2) / 2,
        "lag1_direct": lag1,
        "lag2_direct": lag2,
    }
    return dists, direct, pf.explained_variance_ratio, pf.scores[:, :3]


def rolling_features(panel: np.ndarray, window: int, stride: int):
    """Return features in strictly trailing windows at exclusive end dates."""
    ends = np.arange(window, panel.shape[1] + 1, stride)
    d_store, s_store, variance, pcs = {}, {}, [], []
    for end in ends:
        d, s, eig, coordinates = window_features(panel[:, end - window:end])
        if not d_store:
            d_store = {name: [] for name in d}
            s_store = {name: [] for name in s}
        for name in d:
            d_store[name].append(d[name])
        for name in s:
            n = s[name].shape[0]
            s_store[name].append(s[name][np.triu_indices(n, 1)])
        variance.append(eig)
        pcs.append(coordinates)
    return (
        ends,
        {name: np.stack(values) for name, values in d_store.items()},
        {name: np.stack(values) for name, values in s_store.items()},
        np.stack(variance),
        np.stack(pcs),
    )


def evaluate(panel: np.ndarray, *, boundary: int, window: int, stride: int, has_change: bool):
    """Score pre/post separation, while retaining an unchanged-factor negative control."""
    ends, distances, direct, variance, pcs = rolling_features(panel, window, stride)
    cal = ends <= boundary // 2
    pre = (ends > boundary // 2) & (ends <= boundary)
    post = ends >= boundary + window
    evaluation = pre | post
    if min(cal.sum(), pre.sum(), post.sum()) < 2:
        raise ValueError("Insufficient separated calibration / pre / post windows")
    labels = post[evaluation].astype(int)
    scores, eigengaps, bandwidths, edges_exemplar, eigen_exemplar = {}, {}, {}, {}, {}
    for name, values in distances.items():
        edge, spectral, gap, scale = rolling_graph_features(values, cal, k=3)
        bandwidths[name] = scale
        scores[name + "_graph_edges"] = reference_feature_score(edge, cal)
        scores[name + "_graph_spectrum"] = reference_feature_score(spectral, cal)
        eigengaps[name] = gap
        edges_exemplar[name] = edge
        eigen_exemplar[name] = spectral
    for name, features in direct.items():
        scores[name] = reference_feature_score(features, cal)
    # Spectrum/dispersion is node-permutation-invariant; sign-indeterminate
    # FPCA scores are not compared between windows.
    scores["fpca_explained_spectrum"] = reference_feature_score(variance, cal)
    row = {}
    for name, values in scores.items():
        row[name] = {
            "auroc": float(roc_auc_score(labels, values[evaluation])) if has_change else None,
            "mean_pre": float(np.mean(values[pre])),
            "mean_post": float(np.mean(values[post])),
            # Only descriptive diagnostic: a 95th percentile from five or more
            # autocorrelated calibration windows is NOT a calibrated test.
            "post_threshold_fraction": float(
                np.mean(values[post] > np.quantile(values[cal], 0.95))
            ),
        }
    return row, {
        "ends": ends, "scores": scores, "eigengaps": eigengaps, "bandwidths": bandwidths,
        "distances": distances, "edges": edges_exemplar, "spectra": eigen_exemplar,
        "explained": variance, "pc_scores": pcs,
        "masks": {"cal": cal, "pre": pre, "post": post},
    }


def make_panel(seed: int, dgp: str, length: int, boundary: int):
    """Three stationary-variance synthetic cases; null has no AR coefficient change."""
    parameters = {"ar1_switch": (0.75, -0.65),
                  "ar2_switch": (0.75, -0.65), "ar1_null": (0.7, 0.7)}
    before, after = parameters[dgp]
    if dgp != "ar2_switch":
        panel, _, groups = simulate_market(SimulationSettings(
            seed=seed, n_assets=12, n_groups=3, n_steps=length, boundary=boundary,
            group_strength=0.7, phi_before=before, phi_after=after,
        ))
        return panel, groups
    rng = np.random.default_rng(seed)
    factors = np.empty((3, length))
    for start, stop, phi in ((0, boundary, before), (boundary, length, after)):
        factors[:, start:start + 2] = rng.standard_normal((3, 2))
        for t in range(start + 2, stop):
            factors[:, t] = phi * factors[:, t - 2] + np.sqrt(1 - phi**2) * rng.standard_normal(3)
    groups = np.repeat(np.arange(3), 4)
    returns = np.sqrt(0.7) * factors[groups] + np.sqrt(0.3) * rng.standard_normal((12, length))
    return returns, groups


def save_plot(fig, folder: Path, name: str):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.svg")
    plt.close(fig)


def figures(folder: Path, agg: list, exemplars: dict, boundary: int, window: int):
    """Produce actual plots from archived data, never fabricated illustrations."""
    chosen = [
        "pearson_direct", "lag1_direct", "lag2_direct",
        "pearson_graph_edges", "lag1_graph_edges", "lag2_graph_edges",
        "full_pls_h1_graph_edges", "fpca_pls_k3_graph_edges",
        "fpca_pls_k3_graph_spectrum", "fpca_explained_spectrum",
    ]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for offset, dgp in enumerate(("ar1_switch", "ar2_switch")):
        ys = [next(r["mean_auroc"] for r in agg
                   if r["dgp"] == dgp and r["method"] == m) for m in chosen]
        ax.plot(np.arange(len(chosen)) + (-0.08 if offset == 0 else 0.08), ys,
                "o-", label=dgp, linewidth=1.1)
    ax.set_xticks(np.arange(len(chosen)), chosen, rotation=50, ha="right", fontsize=8)
    ax.set(xlabel="Predeclared representation and statistic", ylabel="Mean held-out regime AUROC",
           title="EXP-006 | Matched graph and temporal-dependence baselines")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    save_plot(fig, folder, "mean_regime_auroc")

    diag = exemplars["ar2_switch"]["analysis"]
    ends = diag["ends"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    series = ("pearson_graph_edges", "lag2_direct",
              "fpca_pls_k3_graph_edges", "fpca_pls_k3_graph_spectrum")
    for name in series:
        s = diag["scores"][name]
        cal = diag["masks"]["cal"]
        denom = max(float(s[cal].std()), 1e-8)
        ax.plot(ends, (s - s[cal].mean()) / denom, linewidth=1.3, label=name)
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1, label="Known synthetic switch")
    ax.axvspan(boundary, boundary + window, alpha=0.12, color="gray", label="Excluded transition")
    ax.set(xlabel="Trailing-window end index", ylabel="Calibration-normalized deviation (descriptive)",
           title="EXP-006 | Dynamic graph scores on one AR(2) simulation")
    ax.legend(fontsize=7, ncol=2)
    save_plot(fig, folder, "dynamic_network_scores")

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for name in ("pearson", "lag2", "full_pls_h1", "fpca_pls_k3"):
        ax.plot(ends, diag["eigengaps"][name], label=name, linewidth=1.4)
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Trailing-window end index", ylabel="Second normalized Laplacian eigenvalue",
           title="EXP-006 | Algebraic connectivity through simulated regimes")
    ax.legend(fontsize=8)
    save_plot(fig, folder, "laplacian_eigenvalue2")

    explained = diag["explained"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for j in range(3):
        ax.plot(ends, explained[:, j], linewidth=1.4, label=f"Cross-sectional FPCA component {j+1}")
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Trailing-window end index", ylabel="Variance proportion", ylim=(0, 1),
           title="EXP-006 | Explained Hilbert-space variance")
    ax.legend(fontsize=8)
    save_plot(fig, folder, "functional_pca_explained_variance")

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.4))
    pc = diag["pc_scores"]
    positions = [np.flatnonzero(diag["masks"]["pre"])[-1], np.flatnonzero(diag["masks"]["post"])[0]]
    groups = exemplars["ar2_switch"]["groups"]
    for ax, idx, tag in zip(axs, positions, ("Before", "After")):
        ax.scatter(pc[idx, :, 0], pc[idx, :, 1], c=groups,
                   cmap="tab10", s=65, edgecolors="k", linewidth=0.3)
        for asset, (x, y) in enumerate(pc[idx, :, :2]):
            ax.annotate(str(asset + 1), (x, y), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)
        ax.set(xlabel="Functional PC1 score", ylabel="Functional PC2 score",
               title=f"{tag}: window ending at {ends[idx]}")
    fig.suptitle("EXP-006 | Within-window FPCA coordinates (axes may rotate between windows)")
    save_plot(fig, folder, "asset_functional_pca_maps")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    null = exemplars["ar1_null"]["analysis"]
    for name in ("pearson_graph_edges", "lag1_direct", "fpca_pls_k3_graph_edges"):
        v = null["scores"][name]
        ax.plot(null["ends"], v, label=name)
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1,
               label="Artificial split; NO parameter shift")
    ax.set(xlabel="Trailing-window end index", ylabel="Raw reference-deviation score",
           title="EXP-006 | Negative control: unchanged AR coefficients")
    ax.legend(fontsize=8)
    save_plot(fig, folder, "null_control_scores")


def run(*, publish: bool, quick: bool, seeds: int):
    if publish and quick:
        raise ValueError("Do not archive quick smoke as research evidence")
    if seeds < 1:
        raise ValueError("seeds must be positive")
    length, boundary, window, stride = (288, 144, 48, 12) if quick else (480, 240, 64, 12)
    rows, exemplars = [], {}
    for index in range(seeds):
        seed = 20261022 + index
        for dgp in ("ar1_switch", "ar2_switch", "ar1_null"):
            panel, groups = make_panel(seed, dgp, length, boundary)
            scores, analysis = evaluate(
                panel, boundary=boundary, window=window, stride=stride,
                has_change=dgp != "ar1_null"
            )
            for method, result in scores.items():
                rows.append({"seed": seed, "dgp": dgp, "method": method, **result})
            if dgp not in exemplars:
                exemplars[dgp] = {"groups": groups, "panel": panel, "analysis": analysis}
    aggregated = []
    for dgp in ("ar1_switch", "ar2_switch", "ar1_null"):
        for name in sorted({r["method"] for r in rows}):
            subset = [r for r in rows if r["dgp"] == dgp and r["method"] == name]
            auc = np.array([r["auroc"] for r in subset if r["auroc"] is not None])
            false = np.array([r["post_threshold_fraction"] for r in subset])
            aggregated.append({
                "dgp": dgp, "method": name, "mean_auroc": float(auc.mean()) if auc.size else None,
                "sd_seed_auroc": float(auc.std(ddof=1)) if len(auc) > 1 else None,
                "mean_post_threshold_fraction": float(false.mean()),
                "n_independent_seeds": seeds,
            })
    summary = {
        "experiment": EXP, "evidence": "exploratory synthetic, not independently confirmed",
        "n_independent_seeds": seeds,
        "config": {
            "asset_count": 12, "n_steps": length, "boundary": boundary, "window": window,
            "stride": stride, "legendre_degree": 7, "pls_difference_order": 1,
            "fixed_guerrero_s": 0.50, "hilbert_derivative_weight": 0.01,
            "fpca_components": [2, 3, 5], "graph_k": 3,
            "bandwidth": ("median positive pairwise distance among calibration "
                          "windows only, per method"),
            "reference": "first half of pre-switch window endpoints",
            "comparison": "pre vs post excluding boundary overlap",
            "notes": ("cross-sectional window-local FPCA, not dynamic "
                      "long-run-covariance FPCA; no label-based tuning"),
        },
        "aggregate": aggregated,
        "limits": [
            "Matching lag-one and lag-two correlation baselines detect "
            "simulated AR changes directly.",
            "No market prices, returns forecasting, or profit evaluation.",
            "Overlapping windows correlated; independent unit is "
            "simulation seed, not each rolling window.",
            "Negative-control exceedance quantile uses few overlapping "
            "calibration windows; not a nominal false-alarm probability.",
            "Window-specific PCA axes may rotate: compare distances/eigenvalues, "
            "not aligned principal-score coordinates.",
            "Model selection from these displayed results would require "
            "separate independent evaluation seeds.",
            "A static graph spectrum is a summary, not an estimator of causal "
            "relationships or conditional dependence.",
        ],
    }
    if not publish:
        quick_subset = [r for r in aggregated if r["method"] in (
            "pearson_graph_edges", "lag2_direct", "fpca_pls_k3_graph_spectrum"
        )]
        print(json.dumps({"n_rows": len(rows), "subset": quick_subset,
                          "config": summary["config"]}, indent=2))
        return summary
    rid = "run_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + f"_n{seeds}"
    result_dir = ROOT / "market_geometry" / "results" / EXP / "runs" / rid
    fig_dir = ROOT / "market_geometry" / "figures" / EXP / "runs" / rid
    result_dir.mkdir(parents=True, exist_ok=False)
    fig_dir.mkdir(parents=True, exist_ok=False)

    def csv_save(filename: str, items: list[dict]):
        with (result_dir / filename).open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)

    csv_save("seed_method_metrics.csv", rows)
    csv_save("mean_metrics.csv", aggregated)
    exemplar = exemplars["ar2_switch"]["analysis"]
    np.savez_compressed(
        result_dir / "ar2_example_timeseries.npz",
        ends=exemplar["ends"], synthetic_group_labels=exemplars["ar2_switch"]["groups"],
        returns=exemplars["ar2_switch"]["panel"],
        fpca_explained=exemplar["explained"], fpca_scores_window=exemplar["pc_scores"],
        laplacian_fpca=exemplar["spectra"]["fpca_pls_k3"],
        laplacian_pearson=exemplar["spectra"]["pearson"],
        graph_edges_fpca=exemplar["edges"]["fpca_pls_k3"],
        score_fpca=exemplar["scores"]["fpca_pls_k3_graph_edges"],
        score_pearson=exemplar["scores"]["pearson_graph_edges"],
        score_lag2=exemplar["scores"]["lag2_direct"],
    )
    summary["run_id"] = rid
    summary["software"] = {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures(fig_dir, aggregated, exemplars, boundary, window)
    files = sorted(list(result_dir.glob("*")) + list(fig_dir.glob("*")))
    manifest = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    (result_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    # Append only after files actually exist. Never remove previous journal evidence.
    figure_names = [
        "mean_regime_auroc", "dynamic_network_scores", "laplacian_eigenvalue2",
        "functional_pca_explained_variance", "asset_functional_pca_maps",
        "null_control_scores",
    ]
    rel = [f"../../figures/{EXP}/runs/{rid}/{name}.png" for name in figure_names]
    block = f"\n## Actual archived run: `{rid}`\n\n"
    block += f"- Independent simulation seeds: **{seeds}**. Synthetic examples only.\n"
    block += f"- Full metrics: [aggregate CSV](../../results/{EXP}/runs/{rid}/mean_metrics.csv); "
    block += f"[per-seed CSV](../../results/{EXP}/runs/{rid}/seed_method_metrics.csv); "
    block += f"[manifest](../../results/{EXP}/runs/{rid}/sha256_manifest.json).\n"
    for name, path in zip(figure_names, rel):
        block += f"\n![{name.replace('_', ' ')}]({path})\n"
    NOTE.write_text(NOTE.read_text(encoding="utf-8") + block, encoding="utf-8")
    print(f"ARCHIVED {rid} ({len(rows)} seed-method rows) in {result_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    run(publish=args.publish, quick=args.quick, seeds=args.seeds)
