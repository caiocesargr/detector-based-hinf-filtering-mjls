"""Reproduce Figure 6 using the shared-path Figure 5 Example 2 experiment.

The paper calls the gray area “variance” but does not specify the exact band
construction. Here it is the pointwise mean squared error plus/minus one
sample standard deviation (ddof=1), with the lower edge clipped at zero.
For a single path, the undefined sample standard deviation is represented
by zero (a collapsed band).

Reuse a matching results/data/figure5_example2_uav.npz when available;
otherwise call Figure 5's Monte Carlo runner with computed filters by default,
T=50, epsilon=0.0075 and its default_rng(seed) convention. The primary error
is (z - zhat)**2, where z=x4+F_theta*w, not the pitch-angle plotting error.
--algorithm1-source published uses equations (31)--(34) only as a reference
reproduction attempt: these coefficients did not reproduce the reported
Monte Carlo behavior with the other published Example 2 data.
Run after pip install -e .: python scripts/figure6_example2_errors.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detector_hinf.plotting import (
    FIGURE_STYLE, SAMPLE_SD_LABEL, SQUARED_ERROR_LABEL, TIME_LABEL, panel_titles,
)

from detector_hinf.models import example2_published_algorithm1_filter, example2_uav_system
from figure5_example2_uav import RESULTS_DIR, run_monte_carlo


def load_experiment(paths=1000, points=2001, seed=12345, results_dir=RESULTS_DIR,
                    algorithm1_source="computed"):
    """Reuse compatible Figure 5 trajectories, including a requested path prefix."""
    if paths < 1 or points < 2 or seed < 0:
        raise ValueError("Require paths >= 1, points >= 2 and seed >= 0.")
    if algorithm1_source not in ("computed", "published"):
        raise ValueError("algorithm1_source must be computed or published.")
    cache = Path(results_dir) / "data" / "figure5_example2_uav.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            matches = (
                int(saved["paths"]) >= paths and int(saved["points"]) == points
                and int(saved["seed"]) == seed and float(saved["T"]) == 50.
                and float(saved["epsilon"]) == 0.0075
                and str(saved["algorithm1_source"]) == algorithm1_source
                and np.array_equal(saved["t"], np.linspace(0., 50., points))
                and (algorithm1_source == "computed" or
                     all(np.array_equal(saved[f"algorithm1_{key}"], np.stack(value))
                        for key, value in example2_published_algorithm1_filter().items()
                         if key in ("Ahat", "Bhat", "Lhat", "Ehat")))
            )
            if matches:
                data = {key: saved[key] for key in saved.files}
                for key in ("x4", "zhat_theorem1", "zhat_algorithm1"):
                    data[key] = data[key][:paths]
                data["path_offsets"] = data["path_offsets"][:paths + 1]
                end = int(data["path_offsets"][-1])
                for key in ("jump_times", "augmented_states"):
                    data[key] = data[key][:end]
                data["paths"] = paths
                # Cached aggregate statistics may describe more paths.
                data = {key: value for key, value in data.items()
                        if not key.startswith(("mean_", "variance_"))}
                data["trajectory_source"] = str(cache)
                print(f"Reusing {paths} Figure 5 paths from {cache}", flush=True)
                return data
    print("No matching Figure 5 cache; running its shared-path experiment.", flush=True)
    data = run_monte_carlo(paths, points, seed, algorithm1_source=algorithm1_source)
    data["trajectory_source"] = "figure5_example2_uav.run_monte_carlo"
    return data


def compute_errors(experiment):
    """Reconstruct z=x4+F_theta*w using right-continuous saved CTMC modes."""
    data = dict(experiment)
    system, _ = example2_uav_system()
    L, F = system[4:6]
    if not all(np.array_equal(value, [[0., 0., 0., 1.]]) for value in L):
        raise ValueError("Reconstructing z from Figure 5 requires L*x = x4.")
    feedthrough = np.array([value[0, 0] for value in F])
    z = np.empty_like(data["x4"])
    for path, (start, end) in enumerate(zip(data["path_offsets"][:-1],
                                          data["path_offsets"][1:])):
        jumps = data["jump_times"][start:end]
        states = data["augmented_states"][start:end]
        sampled = np.searchsorted(jumps, data["t"], side="right") - 1
        theta = states[sampled] // data["Upsilon"].shape[1]
        z[path] = data["x4"][path] + feedthrough[theta] * data["w"]
    data["z"] = z
    data["statistic"] = "pointwise Monte Carlo mean of (z - zhat)**2"
    data["band_construction"] = (
        'The paper calls the gray area "variance" but does not specify the exact '
        "band construction; this implementation uses mean +/- one sample standard "
        "deviation (ddof=1), clipped below at zero; zero width for one path."
    )
    data["variance_ddof"] = 1 if len(z) > 1 else 0
    for key in ("theorem1", "algorithm1"):
        error_sq = (z - data[f"zhat_{key}"])**2
        if not np.isfinite(error_sq).all():
            raise ValueError(f"Nonfinite squared errors for {key}.")
        mean = error_sq.mean(axis=0)
        std = error_sq.std(axis=0, ddof=1) if len(z) > 1 else np.zeros_like(mean)
        data[f"error_sq_{key}"] = error_sq
        data[f"mean_error_sq_{key}"] = mean
        data[f"std_error_sq_{key}"] = std
        data[f"band_lower_{key}"] = np.maximum(0., mean - std)
        data[f"band_upper_{key}"] = mean + std
    return data


def save_results(data, results_dir=RESULTS_DIR):
    data_dir, figures_dir = Path(results_dir) / "data", Path(results_dir) / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "figure6_example2_errors.npz", **data)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True,
                                 constrained_layout=True)
        for ax, title, key in zip(axes, panel_titles(data["algorithm1_source"]),
                                  ("theorem1", "algorithm1")):
            ax.fill_between(data["t"], data[f"band_lower_{key}"],
                            data[f"band_upper_{key}"], color="0.8",
                            label=SAMPLE_SD_LABEL)
            ax.plot(data["t"], data[f"mean_error_sq_{key}"], color="black",
                    linestyle="-", linewidth=1.5, label="Mean squared error")
            ax.axvline(45., color="0.5", linestyle=":", linewidth=1)
            ax.set(title=title, xlabel=TIME_LABEL, xlim=(0., 50.))
            ax.set_ylim(bottom=0.)
            ax.legend(frameon=False)
        axes[0].set_ylabel(SQUARED_ERROR_LABEL)
        stem = figures_dir / "figure6_example2_errors"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    print(f"Saved Figure 6 data and figures under {results_dir}")


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
    save_results(compute_errors(load_experiment(
        args.paths, args.points, args.seed, algorithm1_source=args.algorithm1_source)))


if __name__ == "__main__":
    main()
