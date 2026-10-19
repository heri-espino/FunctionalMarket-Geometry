"""EXP-001: synthetic temporal-memory regime-shift pilot with frozen outputs.

Usage: python market_geometry/code/001-temporal_regime_pilot/run.py --publish
       python market_geometry/code/001-temporal_regime_pilot/run.py --quick
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score

plt.switch_backend("Agg")

# Repository checkout support even without an editable installation.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from market_geometry import (  # noqa: E402
    RollingSettings,
    SimulationSettings,
    SplineSettings,
    reference_deviation,
    rolling_distances,
    simulate_market,
)

EXPERIMENT = "001-temporal_regime_pilot"
NOTE = ROOT / "market_geometry" / "notes" / EXPERIMENT / "README.md"


def _save_figure(fig, figure_dir: Path, basename: str) -> None:
    """Store the same scientific figure as PNG and vector SVG."""
    fig.savefig(figure_dir / f"{basename}.png", dpi=170, bbox_inches="tight")
    fig.savefig(figure_dir / f"{basename}.svg", bbox_inches="tight")
    plt.close(fig)


def run(*, quick: bool, publish: bool, seed: int) -> dict:
    """Run a controlled synthetic demonstration and (optionally) archive it.

    Parameters
    ----------
    quick : bool
        Use a small smoke-test configuration.
    publish : bool
        Create immutable numbered experiment evidence and append linked notes.
    seed : int
        NumPy pseudorandom seed, captured in metadata.

    Returns
    -------
    dict
        Experiment summary, whose statistics describe synthetic data only.
    """
    if quick and publish:
        raise ValueError("--quick is for untracked smoke validation, not publication")
    sim = SimulationSettings(
        seed=seed,
        n_steps=240 if quick else 720,
        boundary=120 if quick else 360,
    )
    cfg = RollingSettings(window=40 if quick else 80, stride=10)
    spline = SplineSettings(n_basis=20 if quick else 36, ridge=0.02, derivative_weight=0.75)
    returns, regimes, groups = simulate_market(sim)
    ends, correlation, functional = rolling_distances(returns, cfg, spline)
    # Only windows entirely before the switch are baseline/calibration candidates.
    calibration = ends <= sim.boundary // 2
    eval_pre = (ends > sim.boundary // 2) & (ends <= sim.boundary)
    # Windows overlapping the regime transition are not classified.
    eval_post = ends >= sim.boundary + cfg.window
    evaluate = eval_pre | eval_post
    if calibration.sum() < 2 or eval_pre.sum() < 2 or eval_post.sum() < 2:
        raise ValueError("Insufficient pre/post windows to evaluate")
    score_corr, reference_corr = reference_deviation(correlation, calibration)
    score_func, reference_func = reference_deviation(functional, calibration)
    labels = eval_post[evaluate].astype(int)
    auroc_corr = roc_auc_score(labels, score_corr[evaluate])
    auroc_func = roc_auc_score(labels, score_func[evaluate])
    baseline_corr = correlation[calibration].mean(axis=0)
    baseline_func = functional[calibration].mean(axis=0)
    after_corr = correlation[eval_post].mean(axis=0)
    after_func = functional[eval_post].mean(axis=0)
    summary = {
        "type": "synthetic_exploratory_pilot_NOT_independent_confirmatory_evidence",
        "description": "AR group factors change persistence, contemporaneous population covariance fixed",
        "seed": seed,
        "simulation": vars(sim),
        "rolling": vars(cfg),
        "splines": vars(spline),
        "n_calibration_windows": int(calibration.sum()),
        "n_held_out_pre_windows": int(eval_pre.sum()),
        "n_held_out_post_windows": int(eval_post.sum()),
        "n_transition_windows_excluded": int((~calibration & ~evaluate).sum()),
        "auroc_correlation": float(auroc_corr),
        "auroc_sobolev": float(auroc_func),
        "caveat": (
            "Single seed; overlapping windows; calibration-only baseline; "
            "post hoc, no formal significance; no financial causality or predictive validation."
        ),
    }
    if quick:
        print(json.dumps(summary, indent=2))
        return summary

    if publish:
        run_id = "run_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + f"_s{seed}"
        result_dir = ROOT / "market_geometry" / "results" / EXPERIMENT / "runs" / run_id
        figure_dir = ROOT / "market_geometry" / "figures" / EXPERIMENT / "runs" / run_id
        result_dir.mkdir(parents=True, exist_ok=False)
        figure_dir.mkdir(parents=True, exist_ok=False)
    else:
        result_dir = ROOT / "artifacts" / "preview" / "results"
        figure_dir = ROOT / "artifacts" / "preview" / "figures"
        result_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        run_id = "local_preview_untracked"

    np.savez_compressed(
        result_dir / "synthetic_panel_and_distances.npz",
        returns=returns,
        known_regime=regimes,
        diagnostic_groups=groups,
        window_ends=ends,
        correlation_distances=correlation,
        sobolev_distances=functional,
        calibration_mask=calibration,
        heldout_pre_mask=eval_pre,
        heldout_post_mask=eval_post,
        baseline_correlation_edges=reference_corr,
        baseline_sobolev_edges=reference_func,
    )
    with (result_dir / "window_scores.csv").open("w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow([
            "window_end_exclusive", "calibration", "heldout_pre", "heldout_post",
            "correlation_network_score", "sobolev_network_score",
        ])
        out.writerows(
            zip(ends, calibration.astype(int), eval_pre.astype(int),
                eval_post.astype(int), score_corr, score_func, strict=True)
        )
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            check=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    summary.update({
        "run_id": run_id,
        "run_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": git_sha,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        },
    })

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(ends, score_corr, label="Correlation-distance network", lw=1.9)
    ax.plot(ends, score_func, label="Spline-Sobolev network", lw=1.9)
    ax.axvline(sim.boundary, ls="--", color="black", lw=1.1, label="True simulation switch")
    ax.axvspan(sim.boundary, sim.boundary + cfg.window, color="gray", alpha=0.15, label="Transition windows excluded")
    ax.set(
        xlabel="Trailing window end (trading-day index)",
        ylabel="RMS change from initial network",
        title="EXP-001 | Synthetic temporal-memory regime switch",
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save_figure(fig, figure_dir, "network_deviation")

    for method, before, after in (
        ("correlation", baseline_corr, after_corr), ("sobolev", baseline_func, after_func)
    ):
        delta = after - before
        max_abs = max(float(np.max(np.abs(delta))), 1e-9)
        fig, ax = plt.subplots(figsize=(6.3, 5))
        hm = ax.imshow(delta, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        fig.colorbar(hm, ax=ax, label="Mean after − mean calibration distance")
        ax.set(
            xlabel="Simulated asset index", ylabel="Simulated asset index",
            title=f"EXP-001 | {method.capitalize()} distance matrix change",
        )
        fig.tight_layout()
        _save_figure(fig, figure_dir, f"distance_delta_{method}")

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i in (0, 1, 4, 8):
        ax.plot(np.arange(returns.shape[1]), returns[i], lw=0.85, alpha=0.78, label=f"Asset {i}")
    ax.axvline(sim.boundary, ls="--", color="black", lw=1)
    ax.set(
        xlabel="Trading-day index", ylabel="Simulated log-return units",
        title="EXP-001 | Artificial returns (not actual stock prices)",
    )
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, figure_dir, "sample_synthetic_returns")

    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    all_outputs = sorted(list(result_dir.iterdir()) + list(figure_dir.iterdir()))
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in all_outputs}
    (result_dir / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if publish:
        images = [
            "network_deviation", "distance_delta_correlation",
            "distance_delta_sobolev", "sample_synthetic_returns",
        ]
        relative_fig_path = "../../figures/" + EXPERIMENT + "/runs/" + run_id
        lines = [
            f"\n### Executed run `{run_id}` (synthetic pilot)\n",
            (
                f"Seed `{seed}`; training-only calibration windows `{calibration.sum()}`; "
                f"held-out pre/post windows `{eval_pre.sum()}`/`{eval_post.sum()}`."
            ),
            (
                f"Exploratory AUROC: correlation `{auroc_corr:.3f}`, "
                f"spline-Sobolev `{auroc_func:.3f}`. "
                "**Single seeded simulation; no statistical claim.**"
            ),
            (
                "The panel is simulated; market factors change temporal persistence "
                "but retain the same population contemporaneous covariance."
            ),
            (
                "The scores summarize distance-matrix deviation; "
                "they are not causal dependence scores or trading forecasts."
            ),
            (
                "Archived numerical results: "
                f"[`summary.json`](../../results/{EXPERIMENT}/runs/{run_id}/summary.json), "
                f"[`window_scores.csv`](../../results/{EXPERIMENT}/runs/{run_id}/window_scores.csv), "
                f"[`NPZ`](../../results/{EXPERIMENT}/runs/{run_id}/synthetic_panel_and_distances.npz), "
                f"[`SHA-256 manifest`](../../results/{EXPERIMENT}/runs/{run_id}/sha256_manifest.json)."
            ),
            "**Figures (generated, not placeholders):**",
        ]
        lines.extend(
            f"![{name.replace('_', ' ').capitalize()}]({relative_fig_path}/{name}.png)"
            for name in images
        )
        with NOTE.open("a", encoding="utf-8") as handle:
            handle.write("\n\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    """Invoke the experiment CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Compute tiny smoke test, no figures or tracked outputs")
    parser.add_argument(
        "--publish", action="store_true",
        help="Archive figures and numerical evidence, append canonical note",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    run(quick=args.quick, publish=args.publish, seed=args.seed)


if __name__ == "__main__":
    main()
