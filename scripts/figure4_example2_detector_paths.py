"""Generate Figure 4's Example 2 Markov-mode and detector sample paths.

Run after installing the repository with pip install -e .:
    python scripts/figure4_example2_detector_paths.py --seed 12345

Saved state indices are zero-based; plotted state labels follow the paper.
The seed makes this realization reproducible, without prescribing the
particular random realization displayed in the paper.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detector_hinf.plotting import FIGURE_STYLE, TIME_LABEL

from detector_hinf.detector import build_augmented_generator
from detector_hinf.models import (
    example2_detector_generators,
    example2_emission_matrix,
    example2_markov_generator,
)
from detector_hinf.simulation import decode_augmented_states, simulate_ctmc


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def generate_path(seed=12345):
    """Return event times, zero-based states, and experiment parameters."""
    epsilon, horizon = 0.0075, 50.0
    Lambda = example2_markov_generator()
    detectors = example2_detector_generators()
    Upsilon = example2_emission_matrix()
    Q = build_augmented_generator(Lambda, detectors, Upsilon, epsilon)
    jump_times, states = simulate_ctmc(
        Q, horizon, initial_state=0, rng=np.random.default_rng(seed),
    )
    theta, detector = decode_augmented_states(states, n_symbols=Upsilon.shape[1])
    return {
        "jump_times": jump_times, "augmented_states": states,
        "theta": theta, "detector_symbol": detector,
        "epsilon": epsilon, "T": horizon, "seed": seed,
        "theta0": 0, "detector0": 0,
        "Lambda": Lambda, "detector_generators": np.stack(detectors),
        "Upsilon": Upsilon, "augmented_generator": Q,
    }


def save_results(data, results_dir):
    """Save the event-driven path and two aligned, right-continuous plots."""
    data_dir, figures_dir = results_dir / "data", results_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "figure4_example2.npz", **data)

    # Extend the final holding interval to T for plotting only. The saved
    # jump_times contain time zero and actual jumps, not an artificial jump at T.
    times = np.append(data["jump_times"], data["T"])
    theta = np.append(data["theta"], data["theta"][-1]) + 1
    detector = np.append(data["detector_symbol"], data["detector_symbol"][-1]) + 1
    with plt.rc_context(FIGURE_STYLE):
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(8.0, 4.5), constrained_layout=True,
        )
        ax_top.step(times, theta, where="post", color="black", linewidth=0.9)
        ax_top.set_ylabel(r"$\theta(t)$")
        ax_top.set_yticks([1, 2])
        ax_top.set_ylim(0.8, 2.2)
        ax_bottom.step(times, detector, where="post", color="black", linewidth=0.9)
        ax_bottom.set_ylabel(r"$\hat{\theta}^{\varepsilon}(t)$")
        ax_bottom.set_yticks([1, 2, 3])
        ax_bottom.set_ylim(0.8, 3.2)
        ax_bottom.set_xlabel(TIME_LABEL)
        for ax in (ax_top, ax_bottom):
            ax.set_xlim(0.0, data["T"])
            ax.grid(True, color="0.9", linewidth=0.6)
        stem = figures_dir / "figure4_example2_detector_paths"
        fig.savefig(stem.with_suffix(".png"))
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)
    print(f"Saved {len(data['jump_times']) - 1} jumps with seed {data['seed']}.")
    print(f"Saved data to {data_dir / 'figure4_example2.npz'}")
    print(f"Saved PNG and PDF figures to {figures_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed (default: 12345).")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    save_results(generate_path(args.seed), RESULTS_DIR)


if __name__ == "__main__":
    main()
