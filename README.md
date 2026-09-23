# Detector-based H-infinity filtering for Markov jump linear systems

This repository will provide a Python implementation to reproduce the numerical
results of the paper **“Detector-based approach for H-infinity filtering of Markov
jump linear systems with partial mode information.”** The goal is to reproduce
the paper's numerical examples, figures, and tables in a transparent workflow.

The project currently contains the initial repository structure. The mathematical
results, optimization routines, and numerical experiments have not yet been
implemented.

CVXPY will be used to formulate the linear matrix inequality (LMI) optimization
problems, with MOSEK as the solver. NumPy and SciPy will support numerical
computations, and Matplotlib will be used for plotting.

## Repository structure

```text
src/detector_hinf/   Python package scaffold
    models.py       System models
    detector.py     Mode detector
    lmi.py          LMI formulations
    algorithm1.py   Implementation of the paper's Algorithm 1
    simulation.py   Numerical simulations
    plotting.py     Plotting utilities
scripts/            Entry points for reproducing Figures 2–6 and Table 2
tests/              Test placeholders for models, detectors, and LMIs
docs/               Documentation directory (currently empty)
results/
    figures/        Directory for generated figures (currently empty)
    tables/         Directory for generated tables (currently empty)
requirements.txt    Python dependencies
CITATION.cff        Citation metadata placeholder
```

The module and script descriptions above indicate their intended roles; the
files are currently placeholders.

## Installation

Clone or download this repository, then open a terminal in its root directory.
Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell), use:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies inside the activated environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include CVXPY and the MOSEK Python package. A valid MOSEK license
must also be configured before solving optimization problems with MOSEK.

This setup installs the dependencies for future development. Reproduction
commands will be documented once the numerical routines are implemented.

To leave the virtual environment, run `deactivate`.
