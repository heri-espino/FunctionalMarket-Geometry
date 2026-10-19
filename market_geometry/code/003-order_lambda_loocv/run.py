"""EXP-003: fair nested LOO for difference orders and Gaussian lambda ML.

Examples:
    python market_geometry/code/003-order_lambda_loocv/run.py --quick
    python market_geometry/code/003-order_lambda_loocv/run.py --publish --seeds 3

Unlike an in-sample LOO-minimized score, the reported OUTER LOOCV RMSE
uses a re-selected lambda on the other n-1 observations each time.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from market_geometry.guerrero import fit_panel  # noqa: E402
from market_geometry.order_selection import (  # noqa: E402
    OrderStudyConfig, nested_loo_comparison,
)

ID = "003-order_lambda_loocv"
METHODS = ("fixed", "inner_loocv", "inner_gcv", "gaussian_ml")
SCENARIOS = ("low_frequency", "oscillatory", "slope_break", "curved")
NOISES = ("iid", "ar1")


def generate(seed: int, n: int, scenario: str, noise: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one simulated return process and its known latent conditional trend."""
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
            error[k] = .6 * error[k-1] + eps[k]
    else:
        raise ValueError("Unknown noise type")
    return trend + error, trend


def series_run(seed: int, n: int, cfg: OrderStudyConfig) -> tuple[list[dict], dict]:
    """Return one independent experiment batch; no oracle truth in fitting."""
    records = []
    example = {}
    for scenario in SCENARIOS:
        for noise in NOISES:
            # Independent noise draw for each configuration and seed, with stable keys.
            subseed = seed * 100 + SCENARIOS.index(scenario) * 10 + NOISES.index(noise)
            series, latent = generate(subseed, n, scenario, noise)
            results = nested_loo_comparison(series, cfg)
            for (d, method), r in results.items():
                records.append({
                    "seed": seed, "scenario": scenario, "noise": noise,
                    "order": d, "selector": method,
                    "outer_loocv_observed_rmse": r["rmse_observed"],
                    "outer_loo_latent_rmse_diagnostic": float(np.sqrt(np.mean(
                        (r["outer_loo_prediction"] - latent)**2
                    ))),
                    "mean_selected_s": float(np.mean(r["selected_s"])),
                    "median_selected_lambda": float(np.median(r["selected_lambda"])),
                    "n_ml_boundary": r["n_boundary_ml"],
                })
            if scenario == "oscillatory" and noise == "iid":
                example = {"observed": series, "latent": latent,
                           "predictions": {f"d{d}_{method}": r["outer_loo_prediction"]
                                           for (d, method), r in results.items()},
                           "fitted_guerrero": {str(d): fit_panel(series[None, :], .75, d=d,
                                                                  mean_mode="guerrero")[0]
                                               for d in cfg.orders},
                           "fitted_zero": {str(d): fit_panel(series[None, :], .75, d=d,
                                                              mean_mode="zero")[0]
                                           for d in cfg.orders}}
    return records, example


def summarize(rows: list[dict], seeds: int, cfg: OrderStudyConfig, n: int) -> dict:
    """Aggregate by seed (independent simulation replicate), not by held-out point."""
    # Every seed/scenario/noise has 4 orders × 4 selectors; report per-setting means.
    result = {}
    for d in cfg.orders:
        for method in METHODS:
            selected = [x for x in rows if x["order"] == d and x["selector"] == method]
            observed = np.asarray([x["outer_loocv_observed_rmse"] for x in selected])
            latent = np.asarray([x["outer_loo_latent_rmse_diagnostic"] for x in selected])
            # Per-seed averages account for the correlation of 8 scenarios within seed.
            per_seed = [np.mean([x["outer_loocv_observed_rmse"] for x in selected
                                 if x["seed"] == s]) for s in sorted({r["seed"] for r in selected})]
            result[f"d{d}_{method}"] = {
                "mean_observed_rmse": float(observed.mean()),
                "mean_latent_rmse_diagnostic": float(latent.mean()),
                "between_seed_sd_observed_rmse": float(np.std(per_seed, ddof=1))
                if seeds > 1 else None,
                "n_ml_boundary_total": int(sum(x["n_ml_boundary"] for x in selected)),
            }
    return {"status": "exploratory_synthetic_comparison_not_publication_evidence",
            "n_seeds": seeds, "n_observations": n, "orders": list(cfg.orders),
            "smoothness_candidates": list(cfg.s_grid), "fixed_s": cfg.fixed_s,
            "ml_log_lambda_bounds": list(cfg.ml_log_lambda_bounds),
            "design": {"scenarios": list(SCENARIOS), "noise_processes": list(NOISES),
                       "outer_cv": "leave one observation out; hyperparameters nested",
                       "inner_loo": "exact two-deletion PRESS for fixed lambda",
                       "likelihood": "Gaussian profile ML of retained complete differences, mu=0",
                       "data": "synthetic daily-return-style levels, not observed stock prices"},
            "by_order_selector": result,
            "limitations": [
                "LOOCV reconstructs individual observations; it does not measure future forecast RMSE.",
                "Adjacent held-out points and the eight scenarios are dependent within seed.",
                "Synthetic latent trend is supplied only to evaluation, never to model selection.",
                "The likelihood uses zero mean for d-th differences, not Guerrero's observed-mu plug-in.",
                "Gaussian ML assumes iid Gaussian measurement and d-th-difference errors; AR1 noise violates this.",
                "Deleting an observation in ML removes any differences touching it: valid but not fully efficient.",
                "Bounds on ML lambda are prespecified and boundary solutions must be disclosed.",
                "Guerrero (2007) selected lambda by smoothness percentage, not by the presented ML rule.",
                "Comparing orders after inspecting their outer RMSE is exploratory model comparison, not a second unbiased selection of d.",
                "No market regime-change results are inferred from this reconstruction experiment.",
            ]}


def write_plot(fig: plt.Figure, dest: Path) -> None:
    fig.savefig(dest.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run(*, seed: int, seeds: int, n: int, publish: bool) -> dict:
    """Execute and optionally commit immutable local archives (never git-push)."""
    if seeds < 1 or n < 20:
        raise ValueError("Seeds >=1 and n>=20 required")
    cfg = OrderStudyConfig()
    rows = []
    for k in range(seeds):
        new, example = series_run(seed+k, n, cfg)
        rows.extend(new)
        if k == 0:
            first_example = example
    summary = summarize(rows, seeds, cfg, n)
    if not publish:
        print(json.dumps({"status": summary["status"], "n_seeds": seeds, "n": n,
                          "mean_observed_rmse": {k: round(v["mean_observed_rmse"], 4)
                                                 for k, v in summary["by_order_selector"].items()}}, indent=2))
        return summary
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_s{seed}_n{seeds}"
    res = ROOT / "market_geometry" / "results" / ID / "runs" / run_id
    figdir = ROOT / "market_geometry" / "figures" / ID / "runs" / run_id
    res.mkdir(parents=True, exist_ok=False)
    figdir.mkdir(parents=True, exist_ok=False)
    summary.update({"run_id": run_id, "seed_start": seed,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "versions": {"python": platform.python_version(),
                                 "numpy": np.__version__, "scipy": scipy.__version__,
                                 "matplotlib": matplotlib.__version__}})
    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (res / "seed_scenario_level.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (res / "one_panel_loo_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        names = list(first_example["predictions"])
        writer.writerow(["t", "observed", "latent", *names])
        for k in range(n):
            writer.writerow([k, first_example["observed"][k], first_example["latent"][k],
                            *(first_example["predictions"][key][k] for key in names)])

    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.plot(first_example["observed"], linestyle="none", marker=".", alpha=.55,
            label="Observed synthetic returns")
    ax.plot(first_example["latent"], color="black", linewidth=2.3,
            label="Known latent trend (diagnostic)")
    for order in cfg.orders:
        ax.plot(first_example["fitted_zero"][str(order)], label=f"d={order}, mu=0, S=.75")
    ax.set(xlabel="Observation", ylabel="Synthetic return", title="PLS trends by difference order (same S)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    write_plot(fig, figdir / "trend_orders_zero_mean")

    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.plot(first_example["observed"], linestyle="none", marker=".", alpha=.55,
            label="Observed synthetic returns")
    ax.plot(first_example["latent"], color="black", linewidth=2.3,
            label="Latent trend (diagnostic)")
    for order in cfg.orders:
        ax.plot(first_example["fitted_guerrero"][str(order)],
                label=f"d={order}, Guerrero observed-difference mu")
    ax.set(xlabel="Observation", ylabel="Synthetic return",
           title="Guerrero mean plug-in, fixed S=.75 (descriptive, not LOO-selected)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    write_plot(fig, figdir / "guerrero_mean_order_gallery")

    means = np.array([[summary["by_order_selector"][f"d{d}_{method}"]["mean_observed_rmse"]
                       for method in METHODS] for d in cfg.orders])
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    img = ax.imshow(means, cmap="viridis", aspect="auto")
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means[i,j]:.3f}", ha="center", va="center",
                    color="white" if means[i,j] < (means.min()+means.max())/2 else "black")
    ax.set(xticks=range(len(METHODS)), yticks=range(len(cfg.orders)),
           xticklabels=[x.replace("_", "\n") for x in METHODS],
           yticklabels=[f"d={d}" for d in cfg.orders],
           title="Nested outer-LOOCV RMSE (observed series, lower is better)")
    fig.colorbar(img, ax=ax, label="RMSE")
    fig.tight_layout()
    write_plot(fig, figdir / "nested_loocv_rmse")

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for method in METHODS:
        vals = [summary["by_order_selector"][f"d{d}_{method}"]["mean_latent_rmse_diagnostic"]
                for d in cfg.orders]
        ax.plot(cfg.orders, vals, marker="o", label=method)
    ax.set(xlabel="Difference order d", ylabel="RMSE vs simulated latent trend",
           xticks=cfg.orders, title="Oracle evaluation only: latent recovery (not used to select lambda)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    write_plot(fig, figdir / "latent_rmse_diagnostic")

    paths = [*res.glob("*"), *figdir.glob("*")]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (res / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    note = ROOT / "market_geometry" / "notes" / ID / "README.md"
    with note.open("a", encoding="utf-8") as f:
        f.write(f"\n### Completed exploratory run `{run_id}`\n\n")
        f.write(f"{seeds} independent seeds × {len(SCENARIOS)} scenarios × {len(NOISES)} noise designs; ")
        f.write(f"{n} observations per synthetic series. This is exploratory and cannot establish ")
        f.write("which differencing order works best on real stocks.\n\n")
        f.write("| Difference order | Fixed S | Nested inner LOOCV | Inner GCV | Gaussian profile ML |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for d in cfg.orders:
            f.write(f"| d={d} | " + " | ".join(
                f'{summary["by_order_selector"][f"d{d}_{method}"]["mean_observed_rmse"]:.4f}'
                for method in METHODS) + " |\n")
        f.write("\nValues are **nested outer-LOOCV reconstruction RMSE** for noisy observed samples. ")
        f.write("They are not future forecasting RMSE. The Gaussian likelihood was derived ")
        f.write("from the 2007 stochastic specification and is *not* presented as Guerrero's ")
        f.write("own original lambda-selection rule. No method was selected using latent truth.\n\n")
        f.write(f"[Summary](../../results/{ID}/runs/{run_id}/summary.json) · ")
        f.write(f"[Seed-scenario CSV](../../results/{ID}/runs/{run_id}/seed_scenario_level.csv) · ")
        f.write(f"[Held-out predictions](../../results/{ID}/runs/{run_id}/one_panel_loo_predictions.csv) · ")
        f.write(f"[SHA-256 manifest](../../results/{ID}/runs/{run_id}/sha256_manifest.json).\n\n")
        for name in ("trend_orders_zero_mean", "guerrero_mean_order_gallery",
                     "nested_loocv_rmse", "latent_rmse_diagnostic"):
            f.write(f"![Actual experiment plot: {name}](../../figures/{ID}/runs/{run_id}/{name}.png)\n\n")
    print(json.dumps({"run_id": run_id, "n_seed_scenario_rows": len(rows),
                      "mean_outer_loocv_rmse": {k: round(v["mean_observed_rmse"], 4)
                                                for k, v in summary["by_order_selector"].items()}}, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--publish", action="store_true")
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()
    run(seed=args.seed, seeds=args.seeds or (1 if args.quick else 3),
        n=args.n or (26 if args.quick else 48), publish=args.publish)
