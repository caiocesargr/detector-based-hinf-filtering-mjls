"""Reproduce Figure 2 using Theorem 1 and the reported Table 1 benchmark.

Run from an environment with this repository installed (pip install -e .):
    python scripts/figure2_example1_cost_vs_rho.py
"""

import argparse
import csv
from pathlib import Path
import sys

import cvxpy as cp
import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a graphical display.
import matplotlib.pyplot as plt
import numpy as np

from detector_hinf.plotting import FIGURE_STYLE

from detector_hinf.lmi import solve_theorem1_example1


# Table 1, method [29]: a reported benchmark, not a computed result.
MODE_INDEPENDENT_REPORTED = 0.42931
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_sweep(rho_values, csv_path, solver="MOSEK", verbose=False):
    """Solve each rho once and save every result, including failed solves."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rho", "status", "gamma_sq", "gamma"])
        writer.writeheader()
        for rho in rho_values:
            try:
                result = solve_theorem1_example1(float(rho), solver=solver, verbose=verbose)
            except cp.error.SolverError as exc:
                print(f"Solver error at rho = {rho:.2f}: {exc}", file=sys.stderr)
                result = {"status": "solver_error", "gamma_sq": None, "gamma": None}
            row = {"rho": float(rho), **result}
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            gamma_text = "unavailable" if row["gamma"] is None else f"{row['gamma']:.10g}"
            print(f"rho = {rho:.2f} | status = {row['status']} | gamma = {gamma_text}",
                  flush=True)
    return rows


def report_table1_comparison(rows):
    """Print reference errors without modifying results or running a solver."""
    print("Table 1 comparison (relative error = absolute error / reported value):")
    # These reported Theorem 1 costs are used only for this comparison.
    for rho, reported in ((0.0, 0.00025), (0.5, 0.42927)):
        row = next((row for row in rows
                    if np.isclose(row["rho"], rho, rtol=0, atol=1e-12)), None)
        computed = None if row is None else row["gamma"]
        if computed is None or not np.isfinite(computed):
            print(f"rho = {rho:.2f} | computed = unavailable | reported = {reported:.10g}"
                  " | absolute error = unavailable | relative error = unavailable")
            continue
        absolute_error = abs(computed - reported)
        relative_error = absolute_error / abs(reported)
        print(f"rho = {rho:.2f} | computed = {computed:.10g} | reported = {reported:.10g}"
              f" | absolute error = {absolute_error:.10g}"
              f" | relative error = {relative_error:.10g} ({relative_error:.4%})")


def plot_results(rows, figures_dir):
    """Plot computed values unchanged; missing solutions appear as gaps."""
    rho = np.array([row["rho"] for row in rows])
    gamma = np.array([np.nan if row["gamma"] is None else row["gamma"] for row in rows])
    if not np.any(np.isfinite(gamma)):
        raise RuntimeError("No finite Theorem 1 costs were obtained; see the CSV and solver errors.")
    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(FIGURE_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.plot(rho, gamma, color="black", linewidth=1.8, label="Theorem 1")
        ax.axhline(MODE_INDEPENDENT_REPORTED, color="0.45", linestyle="--",
                   linewidth=1.4, label="Mode-independent [29] (reported)")
        # Mark the computed samples only: no replacement by reported costs.
        for reference in (0.0, 0.5):
            selected = np.isclose(rho, reference, rtol=0, atol=1e-12) & np.isfinite(gamma)
            ax.scatter(rho[selected], gamma[selected], s=48, facecolors="white",
                       edgecolors="black", linewidths=1.4, zorder=3, clip_on=False)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(loc="best", frameon=False)
        stem = figures_dir / "figure2_example1_cost_vs_rho"
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".png"))
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=101,
                        help="Odd number of uniformly spaced points, including rho=0.5 (default: 101).")
    parser.add_argument("--solver", default="MOSEK", help="CVXPY solver (default: MOSEK).")
    parser.add_argument("--verbose", action="store_true", help="Show solver output.")
    args = parser.parse_args(argv)
    if args.points < 3 or args.points % 2 == 0:
        parser.error("--points must be an odd integer of at least 3 to include rho=0.5.")
    rho_values = np.linspace(0.0, 1.0, args.points)
    csv_path = RESULTS_DIR / "data" / "figure2_example1.csv"
    rows = run_sweep(rho_values, csv_path, solver=args.solver, verbose=args.verbose)
    report_table1_comparison(rows)
    plot_results(rows, RESULTS_DIR / "figures")
    print(f"Saved numerical results to {csv_path}")
    print(f"Saved PDF and PNG figures to {RESULTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
