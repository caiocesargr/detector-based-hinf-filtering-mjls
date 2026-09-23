"""Compute Example 2 costs and compare with reported Table 2 values.

Run after installing the repository with ``pip install -e .``:
    python scripts/table2_example2_costs.py
"""

import argparse
import csv
from pathlib import Path

from detector_hinf.algorithm1 import solve_algorithm1_example2
from detector_hinf.lmi import solve_theorem1_example2


CSV_PATH = Path(__file__).resolve().parents[1] / "results/data/table2_example2.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="MOSEK", help="CVXPY solver (default: MOSEK).")
    parser.add_argument("--verbose", action="store_true", help="Show solver output.")
    args = parser.parse_args(argv)

    theorem = solve_theorem1_example2(solver=args.solver, verbose=args.verbose)
    print("Algorithm 1 iteration history:")
    # The solver prints every iteration's gamma_bar, gamma and improvement.
    algorithm = solve_algorithm1_example2(solver=args.solver, verbose=args.verbose)
    print(f"Algorithm 1 iterations: {algorithm['iterations']}"
          f" | status: {algorithm['status']}"
          f" | converged: {algorithm['converged']}")
    print(f"Theorem 1 status: {theorem['status']}")

    # Published costs are comparison data only; none enter either optimization.
    references = [
        ("Complete information [30]", "0.0002511", None),
        ("Mode-independent [30]", "1.5450", None),
        ("Theorem 1", "0.1024", theorem),
        ("Algorithm 1", "0.0909", algorithm),
    ]
    rows = []
    print("\nTable 2 comparison (Reported column: published reference values)")
    print(f"{'Scenario':<28} {'Computed':>14} {'Reported':>14}")
    for scenario, reported_text, result in references:
        reported = float(reported_text)
        computed = None if result is None else result["gamma"]
        absolute = None if computed is None else abs(computed - reported)
        relative = None if absolute is None else absolute / abs(reported)
        computed_text = "--" if computed is None else f"{computed:.10g}"
        print(f"{scenario:<28} {computed_text:>14} {reported_text:>14}")
        rows.append({
            "scenario": scenario,
            "status": "reported_only" if result is None else result["status"],
            "computed_gamma": computed,
            "gamma_sq": None if result is None else result["gamma_sq"],
            "reported_gamma": reported_text,
            "absolute_difference": absolute,
            "relative_difference": relative,
            "iterations": None if result is None else result.get("iterations"),
        })

    print("\nLiterature benchmarks [30] are reported, not computed.")
    print("Relative difference = absolute difference / reported gamma.")
    for row in rows[2:]:
        if row["absolute_difference"] is None:
            print(f"{row['scenario']}: differences unavailable (no computed cost).")
        else:
            print(f"{row['scenario']}: absolute difference = {row['absolute_difference']:.10g}"
                  f" | relative difference = {row['relative_difference']:.10g}"
                  f" ({row['relative_difference']:.4%})")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved numerical results to {CSV_PATH}")


if __name__ == "__main__":
    main()
