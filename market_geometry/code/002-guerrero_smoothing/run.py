"""EXP-002: matched synthetic panel comparison of Guerrero and spline smoothing.

Usage:
    python market_geometry/code/002-guerrero_smoothing/run.py --quick
    python market_geometry/code/002-guerrero_smoothing/run.py --publish

--quick never writes scientific evidence. --publish archives immutable output.
"""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from market_geometry.guerrero import (  # noqa: E402
    GuerreroSettings, fit_panel, functional_distance, lag_one_crosscorr, select_s,
)
from market_geometry.metrics import (  # noqa: E402
    SplineSettings, correlation_distance, spline_metric, standardize_windows,
)
from market_geometry.rolling import RollingSettings, reference_deviation  # noqa: E402
from market_geometry.simulation import SimulationSettings, simulate_market  # noqa: E402

ID = "002-guerrero_smoothing"
METHODS = (
    "correlation", "lag1_crosscorr", "spline_sobolev",
    "pls_zero_fixed", "pls_guerrero_fixed",
    "pls_zero_gcv", "pls_guerrero_gcv",
    "pls_zero_forecast_cv", "pls_guerrero_forecast_cv",
)
SELECTORS = (("zero", "fixed"), ("guerrero", "fixed"), ("zero", "gcv"),
             ("guerrero", "gcv"), ("zero", "forecast_cv"), ("guerrero", "forecast_cv"))


def _seed_run(seed: int, *, quick: bool) -> dict:
    sim = SimulationSettings(seed=seed, n_steps=240 if quick else 720,
                             boundary=120 if quick else 360)
    returns, regimes, groups = simulate_market(sim)
    cfg = RollingSettings(window=40 if quick else 80, stride=10)
    pls = GuerreroSettings(cv_window=20 if quick else 40)
    spline = SplineSettings(n_basis=20 if quick else 36, ridge=0.02, derivative_weight=0.75)
    ends = np.arange(cfg.window, sim.n_steps + 1, cfg.stride)
    calc = {k: [] for k in METHODS}
    selection = {k: [] for k in METHODS if k.startswith("pls_")}
    for end in ends:
        raw = returns[:, end-cfg.window:end]
        z = standardize_windows(raw)
        calc["correlation"].append(correlation_distance(z))
        calc["lag1_crosscorr"].append(lag_one_crosscorr(z))
        calc["spline_sobolev"].append(spline_metric(z, spline))
        for mode, rule in SELECTORS:
            method = f"pls_{mode}_{rule}"
            s, _ = select_s(raw, pls, mode, rule)
            selection[method].append(s)
            # Entire fit window is trailing known data at evaluation index end.
            fit = fit_panel(z, s, d=pls.order, mean_mode=mode)
            calc[method].append(functional_distance(fit, pls.derivative_weight))
    calibration = ends <= sim.boundary // 2
    held_pre = (ends > sim.boundary // 2) & (ends <= sim.boundary)
    held_post = ends >= sim.boundary + cfg.window
    selected = held_pre | held_post
    if min(calibration.sum(), held_pre.sum(), held_post.sum()) < 2:
        raise ValueError("Too few pre/post windows")
    labels = held_post[selected].astype(int)
    aucs, scores = {}, {}
    for method in METHODS:
        score, _ = reference_deviation(np.stack(calc[method]), calibration)
        scores[method] = score
        aucs[method] = float(roc_auc_score(labels, score[selected]))
    return {
        "seed": seed, "aucs": aucs, "ends": ends, "scores": scores,
        "s_values": selection, "config": {"simulation": vars(sim), "rolling": vars(cfg),
                                               "pls": vars(pls), "spline": vars(spline)},
        "n_calibration": int(calibration.sum()), "n_eval_pre": int(held_pre.sum()),
        "n_eval_post": int(held_post.sum()),
    }


def _figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=155, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run(*, seed: int, n_seeds: int, quick: bool, publish: bool) -> dict:
    """Run independent draws and optionally archive seed-level matched evidence."""
    if n_seeds < 1 or (quick and publish):
        raise ValueError("Need positive n_seeds; quick and publish are exclusive")
    seeds = [seed + i for i in range(n_seeds)]
    runs = [_seed_run(s, quick=quick) for s in seeds]
    matrix = np.array([[r["aucs"][method] for method in METHODS] for r in runs])
    summary = {
        "status": "exploratory_synthetic_not_confirmatory",
        "seed_start": seed, "n_independent_simulated_panels": n_seeds,
        "methods": list(METHODS), "auroc_by_method": {
            method: {
                "mean": float(matrix[:, j].mean()),
                "sd_across_seeds": (float(matrix[:, j].std(ddof=1)) if n_seeds > 1 else None),
                     "min": float(matrix[:, j].min()), "max": float(matrix[:, j].max())}
            for j, method in enumerate(METHODS)
        },
        "comparison": {
            method: {"mean_paired_auroc_difference_vs_correlation": float(np.mean(matrix[:, j] - matrix[:, 0])),
                     "mean_paired_auroc_difference_vs_lag1": float(np.mean(matrix[:, j] - matrix[:, 1])),
                     "n_seed_wins_vs_lag1": int(np.sum(matrix[:, j] > matrix[:, 1]))}
            for j, method in enumerate(METHODS) if j >= 2
        },
        "config_first_seed": runs[0]["config"],
        "limitations": [
            "Regime boundary is used only to define evaluation regions (oracle evaluation design).",
            "Small number of independent synthetic panels; overlapping windows are not independent replicates.",
            "S selects historical return-forecast fit, NOT regime-detection power; "
            "fairness for regime detection unproved.",
            "GCV is potentially biased by autocorrelation; period stationary factors are synthetic.",
            "Lag-one cross-correlation is an intentionally strong conventional competitor.",
            "Hyperparameter grid and preprocessing not adapted after viewing evaluation results.",
            "No real prices, trading backtests or evidence of causal dependence.",
        ],
    }
    if quick:
        print(json.dumps(summary, indent=2))
        return summary
    if not publish:
        print(json.dumps(summary, indent=2))
        return summary
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_s{seed}_n{n_seeds}"
    result_dir = ROOT / "market_geometry" / "results" / ID / "runs" / run_id
    figure_dir = ROOT / "market_geometry" / "figures" / ID / "runs" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    summary.update({"run_id": run_id, "run_utc": datetime.now(timezone.utc).isoformat(),
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                  "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}})
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (result_dir / "seed_level_auroc.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", *METHODS])
        for i, s in enumerate(seeds):
            writer.writerow([s, *[float(x) for x in matrix[i]]])
    with (result_dir / "window_scores_first_seed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_end_exclusive", *METHODS])
        for i, end in enumerate(runs[0]["ends"]):
            writer.writerow([end, *[runs[0]["scores"][method][i] for method in METHODS]])
    with (result_dir / "selected_s_first_seed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_end_exclusive", *runs[0]["s_values"].keys()])
        for i, end in enumerate(runs[0]["ends"]):
            writer.writerow([end, *[v[i] for v in runs[0]["s_values"].values()]])
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.errorbar(x, matrix.mean(axis=0), yerr=(matrix.std(axis=0, ddof=1) if n_seeds > 1 else None),
                fmt="o", capsize=4, markersize=6, label="Mean ± between-seed SD")
    ax.axhline(.5, color="gray", lw=1, ls=":")
    ax.set(xticks=x, xticklabels=[v.replace("_", "\n") for v in METHODS], ylim=(-0.05, 1.1),
           ylabel="Exploratory AUROC", title="Matched synthetic-regime comparison (seed as unit)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    _figure(fig, figure_dir / "seed_level_comparison")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for method in ("correlation", "lag1_crosscorr", "spline_sobolev", "pls_guerrero_forecast_cv"):
        ax.plot(runs[0]["ends"], runs[0]["scores"][method], label=method.replace("_", " "))
    sim = runs[0]["config"]["simulation"]
    ax.axvline(sim["boundary"], color="black", ls="--", label="True simulated switch")
    ax.set(xlabel="Trailing window end", ylabel="RMS network deviation", title="One seed: unequal network-score scales")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _figure(fig, figure_dir / "network_scores_first_seed")
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for method in ("pls_zero_gcv", "pls_guerrero_gcv", "pls_zero_forecast_cv", "pls_guerrero_forecast_cv"):
        ax.plot(runs[0]["ends"], runs[0]["s_values"][method], label=method.replace("_", " "))
    ax.axvline(sim["boundary"], color="black", ls="--")
    ax.set(xlabel="Trailing window end", ylabel="Guerrero normalized S", ylim=(-.05, 1.05),
           title="Training-only S selected at each trailing window")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _figure(fig, figure_dir / "selected_s_first_seed")
    recorded_paths = [*result_dir.glob("*"), *figure_dir.glob("*")]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in recorded_paths}
    (result_dir / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    note = ROOT / "market_geometry" / "notes" / ID / "README.md"
    with note.open("a", encoding="utf-8") as f:
        f.write(f"\n### Completed exploratory run: `{run_id}`\n\n")
        f.write(f"Synthetic independent seeds: **{n_seeds}** ({seeds[0]}–{seeds[-1]}). ")
        f.write("This is **not** independent financial validation, nor a claim of superiority.\n\n")
        f.write("| Method | Mean AUROC | Between-seed SD |\n|---|---:|---:|\n")
        for method in METHODS:
            row = summary["auroc_by_method"][method]
            sd = f'{row["sd_across_seeds"]:.3f}' if row["sd_across_seeds"] is not None else "n/a"
            f.write(f'| {method} | {row["mean"]:.3f} | {sd} |\n')
        f.write(f"\n[Numeric summary](../../results/{ID}/runs/{run_id}/summary.json) · ")
        f.write(f"[Seed-level CSV](../../results/{ID}/runs/{run_id}/seed_level_auroc.csv) · ")
        f.write(f"[Selected S](../../results/{ID}/runs/{run_id}/selected_s_first_seed.csv) · ")
        f.write(f"[SHA-256 manifest](../../results/{ID}/runs/{run_id}/sha256_manifest.json).\n\n")
        for name, caption in (("seed_level_comparison", "Independent-seed summary"),
                              ("network_scores_first_seed", "One-seed network deviations"),
                              ("selected_s_first_seed", "Selected S trajectories")):
            f.write(f'![{caption}](../../figures/{ID}/runs/{run_id}/{name}.png)\n\n')
        f.write("**Interpretation must consider the signed lag-one comparator.** ")
        f.write("AUROC scores on overlapping windows are descriptive, not independent tests.\n")
    print(json.dumps({"run_id": run_id, "n_seeds": n_seeds,
                      "mean_auroc": {m: round(summary["auroc_by_method"][m]["mean"], 3) for m in METHODS}}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--publish", action="store_true")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    run(seed=args.seed, n_seeds=args.seeds or (2 if args.quick else 12), quick=args.quick, publish=args.publish)
