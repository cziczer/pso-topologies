# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research: application of **PSO (Particle Swarm Optimization)** to **TSP (Travelling Salesman Problem)**.

- Python 3.13, managed with **uv** (`pyproject.toml`)
- **Numba** used for JIT-compiled hot paths (distance matrices, fitness evaluation, etc.)
- Old reference code lives in `D:\old_proj\phd\` — use it as a source for porting logic

## Package structure

```
pso/           # PSO variants and core algorithm
topologies/    # Swarm topology definitions (ring, mesh, tree, torus, …)
data/          # TSP instance loading and preprocessing
experiments/   # Experiment runners and result collection
notebooks/     # Jupyter notebooks for analysis and visualisation
main.py        # PyCharm entry point
```

## Key dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `numba` | JIT compilation of inner loops |
| `scipy` | Statistical utilities |
| `networkx` | Graph/topology construction |
| `tsplib95` | Parsing TSPLIB benchmark instances |
| `matplotlib` | Plotting results |
| `pandas` | Tabular result storage |
| `scikit-posthocs` | Post-hoc statistical tests |

## Development

```bash
# Install all dependencies (creates .venv automatically)
uv sync

# Add a new dependency
uv add <package>

# Run entry point
uv run main.py

# Jupyter
uv run jupyter notebook
```

## Numba usage guidelines

- Decorate compute-heavy functions with `@numba.njit` or `@numba.jit(nopython=True)`
- Distance matrix computation and path-length evaluation are primary candidates
- Avoid Python objects inside `@njit` functions — use plain numpy arrays
- Pre-compile with a small warm-up call when benchmarking
