"""EXP-004 matched method families; CLI: --quick or --publish --seeds 6."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from market_geometry.method_comparison import candidate_bank, compare_nested_loo  # noqa: E402

ID = "004-cross_family_methods"
SCENARIOS = ("low_frequency", "oscillatory", "slope_break", "curved")
NOISES = ("iid", "ar1")


def generate(seed: int, n: int, scenario: str, noise: str):
    """Exactly matched to the EXP-003 synthetic signal/noise DGP."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    if scenario == "low_frequency":
        trend = 0.35 + 0.32 * t + 0.30 * np.sin(2 * np.pi * t)
    elif scenario == "oscillatory":
        trend = 0.10 * t + 0.35 * np.sin(5.5 * np.pi * t) + 0.12 * np.sin(14 * np.pi * t)
    elif scenario == "slope_break":
        trend = 0.4 * t - 1.30 * np.maximum(t - 0.48, 0)
    elif scenario == "curved":
        trend = -0.25 + 1.6 * (t - .5) ** 2 + 0.18 * np.sin(3 * np.pi * t)
    else:
        raise ValueError("Unknown scenario")
    eps = rng.normal(0, 0.25, n)
    if noise == "iid":
        error = eps
    elif noise == "ar1":
        error = np.empty(n)
        error[0] = eps[0] / np.sqrt(1 - .6**2)
        for k in range(1, n):
            error[k] = .6 * error[k - 1] + eps[k]
    else:
        raise ValueError("Unknown noise")
    return trend + error, trend


def experiment(seed: int, seeds: int, n: int):
    """Return paired seed/scenario RMSE; never use latent truth in a selector."""
    rows, example, counts = [], None, Counter()
    for repeat in range(seeds):
        actual = seed + repeat
        for scenario in SCENARIOS:
            for noise in NOISES:
                sub = actual * 100 + SCENARIOS.index(scenario) * 10 + NOISES.index(noise)
                observed, latent = generate(sub, n, scenario, noise)
                results = compare_nested_loo(observed)
                for family, val in results["by_family"].items():
                    prediction = val["outer_loo_prediction"]
                    rows.append({"seed": actual, "scenario": scenario, "noise": noise,
                                 "family": family, "outer_loocv_observed_rmse": val["outer_rmse"],
                                 "oracle_latent_rmse_diagnostic": float(np.sqrt(np.mean((prediction - latent)**2))),
                                 "selected_distinct_configurations": len(set(val["selected_candidates"]))})
                    if family == "All families (nested selection)":
                        counts.update(val["selected_families"])
                if repeat == 0 and scenario == "oscillatory" and noise == "iid":
                    example = (observed, latent, results)
    assert example is not None
    return rows, example, counts


def summarize(rows, seed: int, seeds: int, n: int, counts):
    families = list(dict.fromkeys(r["family"] for r in rows))
    records = {}
    for name in families:
        found = [r for r in rows if r["family"] == name]
        per_seed = [np.mean([r["outer_loocv_observed_rmse"] for r in found if r["seed"] == seed + k])
                    for k in range(seeds)]
        records[name] = {
            "mean_observed_nested_loocv_rmse": float(np.mean([r["outer_loocv_observed_rmse"] for r in found])),
            "mean_latent_oracle_diagnostic_rmse": float(np.mean([r["oracle_latent_rmse_diagnostic"] for r in found])),
            "per_seed_mean_observed_rmse": [float(v) for v in per_seed],
            "between_seed_sd": float(np.std(per_seed, ddof=1)) if seeds > 1 else None,
        }
    return {"status": "exploratory_synthetic_only", "seed_start": seed, "n_seeds": seeds,
            "n_observations": n, "n_candidate_configurations": len(candidate_bank(n)),
            "by_family": records, "all_family_inner_selected_counts": dict(counts),
            "dgp": {"scenarios": list(SCENARIOS), "noise": list(NOISES),
                    "generating_code": "Identical four signals and two noise types to EXP-003"},
            "limitations": [
                "Missing-point reconstruction is NOT chronological forecasting or regime classification.",
                "Known latent functions are used ONLY for diagnostic evaluation, never for selection.",
                "The eight scenarios within each seed are dependent; avoid treating folds as independent.",
                "Family candidate grids differ in breadth. Results are exploratory, not an omnibus hypothesis test.",
                "A post-hoc minimum over each family's test RMSE is NOT an unbiased family-selector estimate.",
                "The separately evaluated all-family selector IS nested within outer LOOCV.",
                "Guerrero's mu plug-in, L1 trend filtering, LOESS, wavelets and Kalman fitting are not yet scored.",
                "No real stocks have been analyzed in EXP-004.",
            ]}


def figure(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run(seed=20261020, seeds=6, n=48, publish=False):
    if seeds < 1 or n < 22:
        raise ValueError("Need seeds>=1, n>=22")
    rows, example, counts = experiment(seed, seeds, n)
    summary = summarize(rows, seed, seeds, n, counts)
    ordered = sorted(summary["by_family"].items(), key=lambda x: x[1]["mean_observed_nested_loocv_rmse"])
    ranked = {k: round(v["mean_observed_nested_loocv_rmse"], 5) for k, v in ordered}
    if not publish:
        print(json.dumps({"n_candidates": summary["n_candidate_configurations"],
                          "n_seeds": seeds, "mean_outer_loocv_rmse": ranked}, indent=2))
        return summary
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ") + f"_s{seed}_n{seeds}"
    results = ROOT / "market_geometry" / "results" / ID / "runs" / run_id
    figures = ROOT / "market_geometry" / "figures" / ID / "runs" / run_id
    results.mkdir(parents=True, exist_ok=False)
    figures.mkdir(parents=True, exist_ok=False)
    summary.update({"run_id": run_id, "versions": {"python": platform.python_version(),
                   "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}})
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (results / "seed_scenario_level.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    obs, latent, first = example
    families = list(first["by_family"])
    with (results / "first_panel_outer_predictions.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["t", "observed", "oracle_latent", *families])
        for j in range(n):
            writer.writerow([j, obs[j], latent[j],
                             *(first["by_family"][f]["outer_loo_prediction"][j] for f in families)])
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [k for k, _ in ordered]
    values = [v["mean_observed_nested_loocv_rmse"] for _, v in ordered]
    ax.barh(names[::-1], values[::-1])
    ax.set(xlabel="Mean outer-LOOCV RMSE, held-out observed samples",
           title="Cross-family synthetic trend reconstruction benchmark")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    figure(fig, figures / "all_families_outer_rmse")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(obs, ".", alpha=0.6, label="Synthetic observed returns")
    ax.plot(latent, "k--", linewidth=2, label="Known latent (evaluation only)")
    for name in ("PLS order 1", "PLS order 2", "B-spline degree 3",
                 "Natural cubic smoothing spline", "RBF kernel ridge",
                 "All families (nested selection)"):
        ax.plot(first["by_family"][name]["outer_loo_prediction"], label=name)
    ax.set(xlabel="Observation", ylabel="Synthetic return",
           title="Outer-held-out predictions (oscillatory + iid noise)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    figure(fig, figures / "outer_predictions_gallery")
    vals = np.asarray([[np.mean([r["outer_loocv_observed_rmse"] for r in rows
                                 if r["family"] == name and r["scenario"] == scenario])
                        for scenario in SCENARIOS] for name in names])
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(vals, aspect="auto")
    ax.set(xticks=range(len(SCENARIOS)), xticklabels=SCENARIOS,
           yticks=range(len(names)), yticklabels=names,
           title="Trend shape versus smoothing family: nested LOOCV RMSE")
    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()
    figure(fig, figures / "scenario_rmse_heatmap")
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [*results.glob("*"), *figures.glob("*")]}
    (results / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    note = ROOT / "market_geometry" / "notes" / ID / "README.md"
    with note.open("a", encoding="utf-8") as out:
        out.write(f"\n### Completed exploratory run `{run_id}`\n\n")
        out.write(f"{seeds} independent seeds × 4 trend shapes × 2 noise processes; "
                  f"n={n} equally-spaced samples. Observed outer LOOCV RMSE, "
                  "with all hyperparameter selection nested.\n\n")
        out.write("| Method family | Outer-LOOCV RMSE |\n|---|---:|\n")
        for family, val in ordered:
            out.write(f"| {family} | {val['mean_observed_nested_loocv_rmse']:.4f} |\n")
        out.write(f"\n[Actual summary](../../results/{ID}/runs/{run_id}/summary.json) · "
                  f"[Seed-scenario scores](../../results/{ID}/runs/{run_id}/seed_scenario_level.csv) · "
                  f"[Held-out predictions](../../results/{ID}/runs/{run_id}/first_panel_outer_predictions.csv) · "
                  f"[SHA256](../../results/{ID}/runs/{run_id}/sha256_manifest.json)\n\n")
        for name in ("all_families_outer_rmse", "outer_predictions_gallery", "scenario_rmse_heatmap"):
            out.write(f"![Generated {name}](../../figures/{ID}/runs/{run_id}/{name}.png)\n\n")
        out.write("**Limitations:** The experiment does NOT evaluate future forecasts, observed market returns, "
                  "or market-regime networks. The oracle latent truth was used only in diagnostics. "
                  "All other proposed method families remain untested.\n")
    print(json.dumps({"run_id": run_id, "outer_loocv_rmse": ranked}, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    choice = ap.add_mutually_exclusive_group(required=True)
    choice.add_argument("--quick", action="store_true")
    choice.add_argument("--publish", action="store_true")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    run(seed=args.seed, seeds=args.seeds if args.publish else 1, n=args.n,
        publish=args.publish)
