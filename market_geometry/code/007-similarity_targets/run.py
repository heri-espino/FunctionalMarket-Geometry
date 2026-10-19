"""EXP-007: controlled market similarity targets and reproducible output.

Run from repository root:
    python market_geometry/code/007-similarity_targets/run.py --quick
    python market_geometry/code/007-similarity_targets/run.py --publish --seeds 100
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from market_geometry.similarity_targets import pairwise_scores, simulate_case  # noqa: E402

EXP = "007-similarity_targets"
CASES = ("level_clone", "shared_factor", "lagged_factor", "volatility_clone")
METHODS = (
    "raw_price_rms", "rebased_price_rms", "return_rms", "pearson",
    "pls_h1", "legendre_h1", "max_lag3_correlation", "volatility_log_ratio",
)


def evaluate(seeds: int, length: int, start_seed: int) -> list[dict]:
    """Retrieve planted pair (0,1) versus independent decoy (0,2)."""
    rows = []
    for scenario in CASES:
        for offset in range(seeds):
            seed = start_seed + offset
            scores = pairwise_scores(simulate_case(scenario, seed, length=length))
            for name in METHODS:
                distances = scores[name]
                d01, d02 = float(distances[0, 1]), float(distances[0, 2])
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "method": name,
                    "target_pair_distance": d01,
                    "decoy_pair_distance": d02,
                    "pair_retrieved": int(d01 < d02),
                    "margin_normalized": (d02 - d01) / max(1e-12, d02 + d01),
                })
    return rows


def report(rows: list[dict]) -> list[dict]:
    """Aggregate seed-level results, keeping scenarios separate."""
    summary = []
    for scenario in CASES:
        for name in METHODS:
            x = [r for r in rows if r["scenario"] == scenario and r["method"] == name]
            success = np.asarray([r["pair_retrieved"] for r in x], dtype=float)
            margin = np.asarray([r["margin_normalized"] for r in x], dtype=float)
            summary.append({
                "scenario": scenario,
                "method": name,
                "n_seeds": len(x),
                "retrieval_rate": float(success.mean()),
                "retrieval_standard_error": (
                    float(success.std(ddof=1) / np.sqrt(len(x)))
                    if len(x) > 1 else float("nan")
                ),
                "mean_normalized_margin": float(margin.mean()),
                "mean_target_distance": float(np.mean([r["target_pair_distance"] for r in x])),
            })
    return summary


def draw(figdir: Path, summary: list[dict], length: int, seed: int) -> None:
    """Save deterministic actual PNG and SVG research figures."""
    case = simulate_case("level_clone", seed, length)
    xx = np.arange(length + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10.3, 7.3), sharex=True)
    for k, label in [(0, "A: start 10,000"), (1, "B: start 1,000")]:
        axes[0].plot(xx, case.prices[k], label=label, linewidth=1.7)
        axes[1].plot(xx, case.prices[k] / case.prices[k, 0], label=label, linewidth=1.7)
    axes[0].set(ylabel="Price", title="Identical proportional evolution, different starting price")
    axes[1].set(
        xlabel="Trading days", ylabel="Price / initial price",
        title="Rebased curves coincide exactly",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figdir / f"price_level_invariance.{suffix}", dpi=190)
    plt.close(fig)

    names = (
        "Pearson", "PLS-Sobolev", "Legendre-Sobolev",
        "Best lag <=3", "Raw-return RMS", "Volatility gap",
    )
    keys = (
        "pearson", "pls_h1", "legendre_h1",
        "max_lag3_correlation", "return_rms", "volatility_log_ratio",
    )
    vals = np.asarray([
        [
            next(
                r["retrieval_rate"] for r in summary
                if r["scenario"] == scenario and r["method"] == method
            )
            for method in keys
        ]
        for scenario in CASES
    ])
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    positions = np.arange(len(CASES))
    width = 0.12
    for j, (label, series) in enumerate(zip(names, vals.T)):
        ax.bar(positions + (j - 2.5) * width, series, width=width, label=label)
    ax.set_xticks(
        positions,
        [
            "Same returns, 10x price", "Shared noisy factor",
            "Exact 2-day lag", "Same shape, 2x volatility",
        ],
    )
    ax.set(
        ylim=(0, 1.05),
        ylabel="Planted-pair retrieval fraction",
        title="Diagnostic retrieval across independently simulated seeds",
    )
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(alpha=0.15, axis="y")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figdir / f"target_retrieval.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    scenario_titles = (
        ("shared_factor", "Contemporaneous factor"),
        ("lagged_factor", "Two-day delayed copy"),
        ("volatility_clone", "Same shape, different risk"),
    )
    for ax, (scenario, title) in zip(axes, scenario_titles):
        panel = simulate_case(scenario, seed, length)
        ax.plot(panel.returns[0], label="Asset A", alpha=0.8)
        ax.plot(panel.returns[1], label="Asset B", alpha=0.8)
        ax.set(title=title, xlabel="Trading day", ylabel="Log return")
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figdir / f"scenario_returns.{suffix}", dpi=190)
    plt.close(fig)


def archive(seeds: int, length: int, start_seed: int) -> Path:
    """Produce an immutable experiment run and append real links to its note."""
    stamp = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S%fZ")
    run_id = f"{stamp}_s{start_seed}_n{seeds}"
    result_dir = ROOT / "market_geometry/results" / EXP / "runs" / run_id
    figure_dir = ROOT / "market_geometry/figures" / EXP / "runs" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)
    observations = evaluate(seeds, length, start_seed)
    summary = report(observations)
    for path, records in (
        (result_dir / "seed_level.csv", observations),
        (result_dir / "summary.csv", summary),
    ):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    draw(figure_dir, summary, length, start_seed)
    configuration = {
        "experiment": EXP, "run_id": run_id, "seed_start": start_seed,
        "n_independent_seeds": seeds, "length": length,
        "cases": CASES, "methods": METHODS,
        "selection": "Fixed in advance; no tuning",
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (result_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8",
    )
    hashes = {}
    for path in sorted([*result_dir.iterdir(), *figure_dir.iterdir()]):
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (result_dir / "sha256_manifest.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8",
    )
    note = ROOT / "market_geometry/notes" / EXP / "README.md"
    with note.open("a", encoding="utf-8") as file:
        file.write(f"\n## Archived run: {run_id}\n\n")
        file.write(
            f"- {seeds} independent seeds, {length} observations per asset; "
            f"start seed {start_seed}. Synthetic iid Gaussian innovations.\n"
        )
        file.write(
            f"- [Seed-level data](../../results/{EXP}/runs/{run_id}/seed_level.csv); "
            f"[aggregated scores](../../results/{EXP}/runs/{run_id}/summary.csv).\n\n"
        )
        for scenario in CASES:
            file.write(
                f"### {scenario}\n\n"
                "| Method | Retrieval fraction | Mean normalized margin |\n"
                "|---|---:|---:|\n"
            )
            for row in summary:
                if row["scenario"] == scenario:
                    file.write(
                        f"| {row['method']} | {row['retrieval_rate']:.3f} | "
                        f"{row['mean_normalized_margin']:.3f} |\n"
                    )
            file.write("\n")
        for picture in (
            "price_level_invariance", "scenario_returns", "target_retrieval",
        ):
            file.write(
                f"![{picture}](../../figures/{EXP}/runs/{run_id}/{picture}.png)\n\n"
            )
    print(f"Published immutable synthetic simulation: {run_id}")
    return result_dir


def main() -> None:
    """Choose an unarchived smoke check or an archived full simulation."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--publish", action="store_true")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=42000)
    args = parser.parse_args()
    if args.seeds < 1 or args.length < 32:
        parser.error("Require positive seed count and length >= 32")
    if args.quick:
        for row in report(evaluate(min(args.seeds, 2), args.length, args.seed_start)):
            if row["method"] in ("pearson", "max_lag3_correlation", "legendre_h1"):
                print(row["scenario"], row["method"], f"retrieval={row['retrieval_rate']:.3f}")
    else:
        archive(args.seeds, args.length, args.seed_start)


if __name__ == "__main__":
    main()
