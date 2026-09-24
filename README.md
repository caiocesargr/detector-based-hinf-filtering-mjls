# Detector-based H-infinity filtering for Markov jump linear systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22938714.svg)](https://doi.org/10.5281/zenodo.22938714)

This repository provides a Python implementation and numerical reproduction of
**“Detector-based approach for H-infinity filtering of Markov jump linear systems
with partial mode information”** by **Caio César Graciani Rodrigues, Marcos Garcia
Todorov, and Marcelo Dutra Fragoso**, published in *IET Control Theory & Applications*.
DOI: [10.1049/iet-cta.2018.5640](https://doi.org/10.1049/iet-cta.2018.5640).

The implementation uses CVXPY and MOSEK for linear matrix inequality (LMI) filter
synthesis, NumPy and SciPy for continuous-time Markov chain and plant/filter
simulation, and Matplotlib for Figures 2–6. It also computes the Theorem 1 and
Algorithm 1 entries of Table 2. Numerical outputs and experiment metadata are
saved alongside the figures so that agreement and discrepancies with the paper
can be examined directly. This is a numerical reconstruction, not a claim of
exact reproduction of all published values or stochastic trajectories.

## Repository structure

```text
src/detector_hinf/   Models, detector mappings, LMI synthesis, Algorithm 1,
                    event-driven simulation, and shared plotting typography
scripts/            Command-line entry points for Figures 2–6 and Table 2
tests/              Model, detector, LMI, algorithm, and simulation tests
results/data/       CSV costs and experiment metadata; NPZ Monte Carlo archives
                    are generated locally by reproduction scripts and ignored by Git
results/figures/    Generated PNG/PDF figures; development/ contains small-run views
CITATION.cff        Machine-readable software and preferred paper citation metadata
```

## Installation

From the repository root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the package and its numerical dependencies. **MOSEK requires a
valid license** configured for the local installation. The license file is not
part of this repository. Figure 4 only simulates a sample path; synthesis runs
require the solver license. Figure 6 can reuse a compatible Figure 5 archive
without solving the synthesis problems again.

## Reproducing the results

Run the following commands from the repository root with the virtual environment
activated. Scripts write to fixed filenames in `results/data/` and
`results/figures/`; quick runs and published-coefficient reference runs replace
the corresponding outputs. Preserve existing outputs separately if needed.
The commands do not automatically write into `results/figures/development/`.

### Quick development runs

These use reduced grids or 100 Monte Carlo paths. Figure 4 and Table 2 have no
separate reduced-size mode.

```bash
# Figure 2: reduced sweep, including rho = 0.5
python scripts/figure2_example1_cost_vs_rho.py --points 11

# Figure 3: Example 1 filter comparison
python scripts/figure3_example1_fast_detector.py --paths 100 --points 1001 --seed 12345

# Figure 4: one Example 2 Markov/detector realization
python scripts/figure4_example2_detector_paths.py --seed 12345

# Table 2: Example 2 synthesis costs
python scripts/table2_example2_costs.py

# Figures 5 and 6: computed filters on shared paths; run Figure 5 first
python scripts/figure5_example2_uav.py --paths 100 --points 1001 --seed 12345 --algorithm1-source computed
python scripts/figure6_example2_errors.py --paths 100 --points 1001 --seed 12345 --algorithm1-source computed
```

### Full reproduction runs

These settings match the locally generated full numerical archives: 101 values of
$\rho$ for Figure 2; 1000 paths and 1001 time points for Figure 3; and 1000 paths
and 2001 time points for Figures 5 and 6.

```bash
python scripts/figure2_example1_cost_vs_rho.py --points 101
python scripts/figure3_example1_fast_detector.py --paths 1000 --points 1001 --seed 12345
python scripts/figure4_example2_detector_paths.py --seed 12345
python scripts/table2_example2_costs.py
python scripts/figure5_example2_uav.py --paths 1000 --points 2001 --seed 12345 --algorithm1-source computed
python scripts/figure6_example2_errors.py --paths 1000 --points 2001 --seed 12345 --algorithm1-source computed
```

Figure 2 requires an odd number of sweep points, at least three. Figure 2 and
Table 2 also accept `--solver` and `--verbose`; their default solver is MOSEK.

Figures 5 and 6 compare **computed Theorem 1** in panel (a) with **computed
Algorithm 1** in panel (b). Both filters use the same Markov/detector realization
for each path. Figure 6 reuses matching Figure 5 data, including a requested
prefix of saved paths, when the grid, seed, and coefficient source agree;
otherwise it invokes the same Monte Carlo runner. Example 2 uses $T=50$ and
$\varepsilon=0.0075$. The dotted line at $t=45$ marks the disturbance cutoff.

Figure 5 shows pointwise Monte Carlo means of the pitch angle $x_4$ and estimate
$\hat z$. Figure 6 instead evaluates the theoretical filtering error for each
path before averaging:

$$
z(t)=x_4(t)+\kappa_{\theta(t)}w(t),\qquad
\mathrm{error}_{\mathrm{sq}}(t)=(z(t)-\hat z(t))^2.
$$

Thus Figure 6 does not use $(x_4-\hat z)^2$ or the squared difference of mean
trajectories. The disturbance is $w(t)=0.5\sin(0.2t)$ through $t=45$ inclusive,
and zero afterwards.

### Published-coefficient reference option

Both Example 2 figure scripts accept `--algorithm1-source published` to use the
Algorithm 1 coefficients in equations (31)–(34):

```bash
python scripts/figure5_example2_uav.py --paths 100 --points 1001 --seed 12345 --algorithm1-source published
python scripts/figure6_example2_errors.py --paths 100 --points 1001 --seed 12345 --algorithm1-source published
```

These coefficients are retained for diagnostic/reference reproduction attempts.
They did not reproduce the published Monte Carlo behaviour when combined with
the other published Example 2 data as reconstructed here. The default remains
`computed`; the published coefficients do not enter the optimization routines.

## Reproduction summary

The values below are taken from
[`figure2_example1.csv`](results/data/figure2_example1.csv),
[`table2_example2.csv`](results/data/table2_example2.csv), and NPZ archives
generated locally by the reproduction scripts. These NPZ outputs are ignored
by Git and are not version-controlled or distributed in this repository.
“Reported” denotes paper values, not independently recomputed
benchmarks. Optimization costs are $\gamma$, not $\gamma^2$.

| Result | Reported in the paper | Recomputed/stored in this repository | Assessment |
|---|---|---|---|
| Example 1, Theorem 1, $\rho=0$ | $\gamma=0.00025$ | $0.0011947836$ | Small absolute cost, but substantial relative discrepancy; not exact reproduction |
| Example 1, Theorem 1, $\rho=0.5$ | $\gamma=0.42927$ | $0.4296819217$ | Close numerical reproduction: about 0.096% relative difference |
| Example 1, mode-independent benchmark [29] | $\gamma=0.42931$ | Plotted as a reported reference only | Not recomputed |
| Example 2, Theorem 1 | $\gamma=0.1024$ | $0.1136611299$ | About 11.00% higher; discrepancy remains |
| Example 2, Algorithm 1 | $\gamma=0.0909$; two iterations | $0.0930057072$; two iterations | Cost about 2.32% higher; iteration count agrees exactly |
| Example 2, complete-information / mode-independent benchmarks [30] | $0.0002511$ / $1.5450$ | Stored as reported references only | Not recomputed |
| Figure 6, time-average mean squared error on shared paths | No scalar reference used for this statistic | Theorem 1: $0.0296655975$; Algorithm 1: $0.0254671468$ | Computed Algorithm 1 reduces this error by 14.15% |

**Example 1.** Figure 2 closely reproduces the reported cost near $\rho=0.5$
and the qualitative cost curve, with the endpoint qualification shown above.
Figure 3 is a **qualitative reproduction**: the paper does not report the
Example 1 filter coefficients, so this repository recovers its own filters.
Its stored experiment uses $\rho=0.2$, $\varepsilon=0.075$, $T=5$, and 1000 paths;
the recovered Theorem 1 cost is $\gamma=0.3624824451$.

**Example 2.** Figure 4 is a **stochastic sample-path reproduction**; its seeded
realization is not expected to match the published realization exactly. The
Theorem 1 result is approximately $0.11365$ in the development comparison
($0.1136611299$ in the current archive), rather than the reported $0.1024$.
An independent direct implementation of the Theorem 1 LMIs was reported during
project development to produce essentially the same $0.11365$, supporting
internal consistency. That development check is not archived as a separate
reproducible result in `results/data/`; it does not resolve the discrepancy
with the paper.

Algorithm 1 converges in two iterations, as reported in the paper, and obtains
approximately $\gamma=0.09301$. Figures 5 and 6 are comparisons of the
recomputed filters, not exact reproductions of the published curves. The error
comparison in the table uses the same 1000 paths for both filters and the
trapezoidal integral of the pointwise mean of $(z-\hat z)^2$ over $[0,50]$,
divided by 50. Improvement in this experiment is distinct from the guaranteed
$H_\infty$ cost and does not imply lower error at every time point.

The rendered results are available as PNG and PDF:
[Figure 2](results/figures/figure2_example1_cost_vs_rho.png),
[Figure 3](results/figures/figure3_example1_fast_detector.png),
[Figure 4](results/figures/figure4_example2_detector_paths.png),
[Figure 5](results/figures/figure5_example2_uav.png), and
[Figure 6](results/figures/figure6_example2_errors.png).

## Numerical notes

- Strict LMIs are implemented using a small positive definiteness margin
  (default $10^{-8}$). Solver tolerances and this margin matter especially
  when the target cost is close to zero.
- CVXPY and a modern MOSEK version may return different feasible or optimal
  filter coefficients from the original MATLAB/MOSEK implementation.
  Dependencies are not version-pinned, so identical results across solver
  versions and platforms are not guaranteed.
- Monte Carlo figures use fixed seeds (default `12345`). Markov jumps are
  simulated in continuous time; the output time grid controls sampling, not
  the random jump process. Integration restarts at jumps and, for Example 2,
  at the disturbance cutoff.
- Figures 3 and 6 use pointwise mean $\pm$ one **sample standard deviation**
  (`ddof=1`), with the lower edge clipped at zero. For Figure 6, this is an
  explicit plotting convention: the paper calls the gray region “variance”
  without specifying its exact construction. The bands are not confidence
  intervals; a one-path development run uses a collapsed band.
- Figure 5 uses pointwise Monte Carlo means as an explicit plotting
  convention; the paper does not unambiguously specify that aggregation.
  Plotting uses portable Matplotlib math text and does not require LaTeX.

## Testing

Install the test dependency if necessary, then run the suite:

```bash
pip install pytest
pytest -q
```

Tests cover model matrices, detector partitions and generators, LMI assembly
and filter recovery, Algorithm 1 stopping/failure handling, and event-driven
simulation against independent numerical references. The licensed Algorithm 1
solver integration test is opt-in:

```bash
RUN_MOSEK_TESTS=1 pytest -q
```

Passing these tests establishes implementation checks, not exact agreement with
all published numerical results.

## Citation

**Original research paper (preferred scientific citation)**

Please cite the original paper when using this implementation:

> Caio César Graciani Rodrigues, Marcos Garcia Todorov, and Marcelo Dutra Fragoso.
> “Detector-based approach for H-infinity filtering of Markov jump linear systems
> with partial mode information.” *IET Control Theory & Applications*.
> DOI: [10.1049/iet-cta.2018.5640](https://doi.org/10.1049/iet-cta.2018.5640).

**Software archive**

To cite this software implementation and its archived release:

> C. C. Graciani Rodrigues, M. G. Todorov, and M. D. Fragoso.
> Detector-based H-infinity filtering for Markov jump linear systems, version 1.0.1.
> Zenodo. DOI: [10.5281/zenodo.22938714](https://doi.org/10.5281/zenodo.22938714).

GitHub can use [`CITATION.cff`](CITATION.cff) to provide machine-readable citation
metadata for the software archive and the preferred scientific citation above.

## License

The repository source code is released under the [MIT License](LICENSE).
Third-party dependencies, including MOSEK, remain subject to their own licenses.
