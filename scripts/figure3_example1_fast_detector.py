"""Reproduce Figure 3 using Monte Carlo paths and recovered Theorem 1 filters.

Run after installing the repository with pip install -e .:
    python scripts/figure3_example1_fast_detector.py
For a quick development run:
    python scripts/figure3_example1_fast_detector.py --paths 50 --points 501

The NPZ stores individual scalar-output paths, pointwise Monte Carlo
statistics, recovered filter matrices, and the experiment parameters.
The gray band shows one sample standard deviation, not a confidence interval.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detector_hinf.detector import build_augmented_generator
from detector_hinf.lmi import solve_theorem1_example1
from detector_hinf.models import (
    example1_detector_generators,
    example1_emission_matrix,
    example1_system,
)
from detector_hinf.simulation import simulate_ctmc, simulate_example1_dynamics


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_monte_carlo(paths=1000, points=1001, seed=12345):
    """Solve once, then simulate independent shared-path filter comparisons.

    Individual zhat_a, zhat_b, and error_sq arrays have shape (paths, points).
    Standard deviations use ddof=1, or zero for a single-path development run.
    A seeded NumPy Generator drives only the CTMC; no fixed-grid chain
    approximation is used. Both filters on a path share the plant and CTMC.
    """
    if paths < 1 or points < 2:
        raise ValueError("paths must be at least 1 and points at least 2.")
    rho, epsilon, horizon = 0.2, 0.075, 5.0
    Upsilon = example1_emission_matrix(rho)
    print("Solving Theorem 1 at rho = 0.20 with MOSEK...", flush=True)
    filters = solve_theorem1_example1(rho, return_filter=True)
    if filters["status"] not in ("optimal", "optimal_inaccurate") or filters["gamma"] is None:
        raise RuntimeError(f"Filter design failed: status = {filters['status']}")
    print(f"status = {filters['status']} | gamma = {filters['gamma']:.10g}", flush=True)
    Q = build_augmented_generator(
        example1_system()[-1], example1_detector_generators(), Upsilon, epsilon,
    )
    time_grid = np.linspace(0.0, horizon, points)
    rng = np.random.default_rng(seed)
    zhat_a = np.empty((paths, points))
    zhat_b = np.empty((paths, points))
    error_sq = np.empty((paths, points))
    progress_interval = max(1, paths // 20)
    for path in range(paths):
        # Index 0 encodes theta(0)=0 and detector(0)=0.
        jump_times, states = simulate_ctmc(Q, horizon, initial_state=0, rng=rng)
        trajectories = simulate_example1_dynamics(
            time_grid, jump_times, states, Upsilon, filters,
        )
        a, b = trajectories["zhat_a"], trajectories["zhat_b"]
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise RuntimeError(f"Nonfinite output estimates on path {path + 1}.")
        zhat_a[path] = a[:, 0]
        zhat_b[path] = b[:, 0]
        # Square each path's output difference before Monte Carlo averaging.
        error_sq[path] = np.sum((a - b)**2, axis=1)
        if path == 0 or (path + 1) % progress_interval == 0 or path + 1 == paths:
            print(f"Completed path {path + 1}/{paths}", flush=True)

    mean_error = error_sq.mean(axis=0)
    std_error = error_sq.std(axis=0, ddof=1) if paths > 1 else np.zeros(points)
    return {
        "t": time_grid,
        "zhat_a": zhat_a, "zhat_b": zhat_b, "error_sq": error_sq,
        "mean_zhat_a": zhat_a.mean(axis=0), "mean_zhat_b": zhat_b.mean(axis=0),
        "mean_error_sq": mean_error, "std_error_sq": std_error,
        "error_band_lower": np.maximum(0.0, mean_error - std_error),
        "error_band_upper": mean_error + std_error,
        "rho": rho, "epsilon": epsilon, "T": horizon,
        "paths": paths, "points": points, "seed": seed,
        "theta0": 0, "detector0": 0,
        "x0": np.ones(3), "xhat_a0": np.zeros(3), "xhat_b0": np.zeros(3),
        "w": np.sin(time_grid), "Upsilon": Upsilon, "augmented_generator": Q,
        "solver": "MOSEK", "status": filters["status"],
        "gamma": filters["gamma"], "gamma_sq": filters["gamma_sq"],
        "std_ddof": 1 if paths > 1 else 0,
        **{name: np.stack(filters[name]) for name in ("Ahat", "Bhat", "Lhat", "Ehat")},
    }


def save_results(data, results_dir):
    """Save the numerical experiment and the two-panel Figure 3."""
    data_dir, figures_dir = results_dir / "data", results_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "figure3_example1.npz", **data)
    with plt.rc_context({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "legend.fontsize": 9, "pdf.fonttype": 42, "savefig.dpi": 300,
    }):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
        t = data["t"]
        ax_a.plot(t, data["mean_zhat_a"], color="black", linestyle="-",
                  linewidth=1.6, label=r"$\mathbb{E}[\hat z_a(t)]$ — filter (6)")
        ax_a.plot(t, data["mean_zhat_b"], color="black", linestyle="--",
                  linewidth=1.6, label=r"$\mathbb{E}[\hat z_b(t)]$ — filter (7)")
        ax_a.set_title("(a) Mean filter estimates")
        ax_a.set_ylabel(r"Mean estimate $\hat z(t)$")
        ax_b.fill_between(t, data["error_band_lower"], data["error_band_upper"],
                          color="0.8", label=r"Mean $\pm$ one standard deviation")
        ax_b.plot(t, data["mean_error_sq"], color="black", linestyle="-",
                  linewidth=1.6, label="Mean squared difference")
        ax_b.set_title("(b) Squared difference between filters")
        ax_b.set_ylabel(r"$\|\hat z_a(t)-\hat z_b(t)\|^2$")
        ax_b.set_ylim(bottom=0.0)
        for ax in (ax_a, ax_b):
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0.0, data["T"])
            ax.grid(True, color="0.9", linewidth=0.6)
            ax.legend(loc="best", frameon=False)
        stem = figures_dir / "figure3_example1_fast_detector"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    print(f"Saved data to {data_dir / 'figure3_example1.npz'}")
    print(f"Saved PNG and PDF figures to {figures_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1000, help="Monte Carlo paths (default: 1000).")
    parser.add_argument("--points", type=int, default=1001, help="Output grid points (default: 1001).")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (default: 12345).")
    args = parser.parse_args(argv)
    if args.paths < 1 or args.points < 2 or args.seed < 0:
        parser.error("--paths must be positive, --points at least 2, and --seed nonnegative.")
    data = run_monte_carlo(paths=args.paths, points=args.points, seed=args.seed)
    save_results(data, RESULTS_DIR)


if __name__ == "__main__":
    main()
