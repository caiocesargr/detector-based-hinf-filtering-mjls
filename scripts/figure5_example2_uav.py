"""Reproduce Figure 5 with shared-path Example 2 Monte Carlo simulations.

The paper reports 1000 Monte Carlo paths but does not explicitly describe
whether the plotted trajectories are Monte Carlo means. This implementation
uses pointwise Monte Carlo means for reproducibility.

Run after pip install -e .; for development use --paths 5 --points 501.
The NPZ stores individual pitch/estimate trajectories, means, sample variances,
filter coefficients, and experiment metadata. Both filters are computed by
default. --algorithm1-source published uses equations (31)--(34) only as a
reference reproduction attempt: these coefficients did not reproduce the
reported Monte Carlo behavior with the other published Example 2 data.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detector_hinf.plotting import FIGURE_STYLE, TIME_LABEL, panel_titles

from detector_hinf.algorithm1 import solve_algorithm1_example2
from detector_hinf.detector import build_augmented_generator
from detector_hinf.lmi import solve_theorem1_example2
from detector_hinf.models import (
    example2_detector_generators, example2_emission_matrix,
    example2_markov_generator, example2_published_algorithm1_filter,
)
from detector_hinf.simulation import simulate_ctmc, simulate_example2_dynamics

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_monte_carlo(paths=1000, points=2001, seed=12345, algorithm1_source="computed"):
    """Design filters once, then use exactly the same CTMC path for both."""
    if paths < 1 or points < 2 or seed < 0:
        raise ValueError("Require paths >= 1, points >= 2 and seed >= 0.")
    if algorithm1_source not in ("computed", "published"):
        raise ValueError("algorithm1_source must be computed or published.")
    print("Solving Theorem 1 with MOSEK...", flush=True)
    theorem = solve_theorem1_example2(return_filter=True)
    if theorem["status"] not in ("optimal", "optimal_inaccurate") or theorem["gamma"] is None:
        raise RuntimeError(f"Theorem 1 failed: {theorem['status']}")
    if algorithm1_source == "computed":
        algorithm = solve_algorithm1_example2(return_filter=True)
        if not algorithm["converged"]:
            raise RuntimeError(f"Algorithm 1 did not converge: {algorithm['status']}")
    else:
        # Reference only: the published coefficients did not reproduce the
        # reported Monte Carlo behavior with the other published Example 2 data.
        algorithm = example2_published_algorithm1_filter()
    print(f"Algorithm 1 coefficients: {algorithm1_source}", flush=True)
    horizon, epsilon = 50., 0.0075
    Upsilon = example2_emission_matrix()
    Q = build_augmented_generator(example2_markov_generator(),
                                  example2_detector_generators(), Upsilon, epsilon)
    t = np.linspace(0., horizon, points)
    rng = np.random.default_rng(seed)
    x4, zhat_theorem1, zhat_algorithm1 = [np.empty((paths, points)) for _ in range(3)]
    jump_times, jump_states, offsets = [], [], [0]
    for path in range(paths):
        jumps, states = simulate_ctmc(Q, horizon, initial_state=0, rng=rng)
        # Integrate the same plant equations and forcing along the same CTMC
        # realization. Adaptive step sizes can differ between the two solves.
        a = simulate_example2_dynamics(t, jumps, states, Upsilon, theorem)
        b = simulate_example2_dynamics(t, jumps, states, Upsilon, algorithm)
        for result in (a, b):
            if not all(np.isfinite(value).all() for value in result.values()):
                raise RuntimeError(f"Nonfinite trajectory on path {path + 1}.")
        x4[path] = a["x"][:, 3]  # Pitch angle, not z=x4+F*w.
        zhat_theorem1[path] = a["zhat"][:, 0]
        zhat_algorithm1[path] = b["zhat"][:, 0]
        jump_times.extend(jumps)
        jump_states.extend(states)
        offsets.append(len(jump_times))
        if path == 0 or (path + 1) % max(1, paths // 20) == 0 or path + 1 == paths:
            print(f"Completed path {path + 1}/{paths}", flush=True)
    data = {
        "t": t, "T": horizon, "epsilon": epsilon, "paths": paths,
        "points": points, "seed": seed, "theta0": 0, "detector0": 0,
        "x0": np.ones(4), "xhat0": np.zeros(4),
        "w": np.where(t <= 45., .5 * np.sin(.2 * t), 0.),
        "Upsilon": Upsilon, "augmented_generator": Q,
        "jump_times": np.asarray(jump_times), "augmented_states": np.asarray(jump_states),
        "path_offsets": np.asarray(offsets),
        "algorithm1_source": algorithm1_source, "solver": "MOSEK",
        "theorem1_status": theorem["status"], "theorem1_gamma": theorem["gamma"],
        "theorem1_gamma_sq": theorem["gamma_sq"],
        "algorithm1_status": algorithm.get("status", "reported_coefficients"),
        "statistic": "pointwise Monte Carlo mean", "variance_ddof": 1 if paths > 1 else 0,
    }
    for name, values in (("x4", x4), ("zhat_theorem1", zhat_theorem1),
                         ("zhat_algorithm1", zhat_algorithm1)):
        data[name] = values
        data[f"mean_{name}"] = values.mean(axis=0)
        data[f"variance_{name}"] = values.var(axis=0, ddof=1) if paths > 1 else np.zeros(points)
    for label, filters in (("theorem1", theorem), ("algorithm1", algorithm)):
        for key in ("Ahat", "Bhat", "Lhat", "Ehat"):
            data[f"{label}_{key}"] = np.stack(filters[key])
    if algorithm1_source == "computed":
        for key in ("gamma", "gamma_sq", "iterations"):
            data[f"algorithm1_{key}"] = algorithm[key]
        for key in ("gamma_bar", "gamma"):
            data[f"algorithm1_history_{key}"] = np.array([row[key] for row in algorithm["history"]])
    return data


def save_results(data, results_dir):
    """Save raw trajectories/statistics and two panels with shared axes."""
    data_dir, figures_dir = results_dir / "data", results_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "figure5_example2_uav.npz", **data)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True,
                                 constrained_layout=True)
        for ax, label, key in zip(axes, panel_titles(data["algorithm1_source"]),
                                  ("theorem1", "algorithm1")):
            ax.plot(data["t"], data["mean_x4"], color="black", linestyle="-",
                    linewidth=1.5, label=r"Mean $x_4(t)$")
            ax.plot(data["t"], data[f"mean_zhat_{key}"], color="red", linestyle="-",
                    linewidth=1.5, label=r"Mean $\hat z(t)$")
            ax.axvline(45., color="0.5", linestyle=":", linewidth=1)
            ax.set_title(label)
            ax.set_xlabel(TIME_LABEL)
            ax.set_xlim(0., 50.)
            ax.grid(True, color="0.9", linewidth=.6)
            ax.legend(frameon=False)
        axes[0].set_ylabel("Pitch angle / estimate")
        stem = figures_dir / "figure5_example2_uav"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    print(f"Saved Figure 5 data and figures under {results_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--points", type=int, default=2001)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--algorithm1-source", choices=("computed", "published"), default="computed",
                        help="Algorithm 1 coefficients (default: computed; published is reference only)")
    args = parser.parse_args(argv)
    if args.paths < 1 or args.points < 2 or args.seed < 0:
        parser.error("Require --paths >= 1, --points >= 2 and --seed >= 0.")
    data = run_monte_carlo(args.paths, args.points, args.seed, args.algorithm1_source)
    save_results(data, RESULTS_DIR)


if __name__ == "__main__":
    main()
