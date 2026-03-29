# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research: application of **PSO (Particle Swarm Optimization)** to **TSP (Travelling Salesman Problem)**.

- Python 3.13, managed with **uv** (`pyproject.toml`)
- **Numba** used for JIT-compiled hot paths (distance matrix already JIT-compiled; path evaluation and operators are candidates for future optimisation)
- Old reference code is the **primary source** for porting logic with improvements — path: `~/Downloads/old_proj/phd/_deprecated/PSO_old/`

## Package structure

```
pso/
  base.py              # PSOBase — swarm state, greedy init, main loop, topology/operator hooks
  factory.py           # AlgorithmFactory — parses name strings, assembles composed classes
  operators/
    default.py         # DefaultOperatorsMixin — order crossover (OX) + swap mutation
topologies/
  base.py              # TopologyMixin ABC — cooperative __init__, build_topology hook
  global_.py           # GlobalTopologyMixin — no topology, all particles share global best
  ring.py              # RingTopologyMixin
  tree.py              # TreeTopologyMixin
  mesh.py              # MeshTopologyMixin
  torus.py             # TorusTopologyMixin
  free_scale.py        # FreeScaleTopologyMixin — Barabási-Albert scale-free graph
data/
  loader.py            # TSPInstance dataclass + InstanceLoader ABC
  tsplib/
    reader.py          # TSPLibReader — parses TSPLIB95 format, Numba JIT distance matrix
    *.tsp              # 47 TSPLIB benchmark instances
experiments/
  result.py            # RunResult, ExperimentResult dataclasses
  runner.py            # ExperimentRunner — runs N repetitions, collects results
  serialiser.py        # ResultSerialiser — JSON save/load with aggregate statistics
notebooks/             # Jupyter notebooks for analysis and visualisation (future)
main.py                # CLI entry point (argparse)
```

## Architecture: mixin composition

PSO variants are composed via **cooperative multiple inheritance**. The factory assembles a class dynamically:

```
type(name, (OperatorMixin, TopologyMixin, PSOBase), {})
```

MRO: `OperatorMixin → TopologyMixin → PSOBase`. The main loop lives only in `PSOBase.run()` and calls two hooks:
- `get_neighbourhood_best(i)` — provided by topology mixin
- `crossover(current, guide)` / `mutate(path)` — provided by operator mixin

**Algorithm name format**: `PSO[-OperatorVariant][-Topology]`
- `"PSO"` → global topology + default operators
- `"PSO-Ring"` → ring topology + default operators
- `"PSO-Operators1-Ring"` → ring topology + Operators1 variant *(future — add `Operators1Mixin` to `pso/operators/` and register in `_OPERATOR_MAP` in `pso/factory.py`)*

## Key dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical arrays |
| `numba` | JIT compilation of inner loops |
| `scipy` | Statistical utilities |
| `networkx` | Graph/topology construction (FreeScale) |
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

# Run experiment via CLI
uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 30

# Jupyter
uv run jupyter notebook
```

## Adding a new topology

1. Create `topologies/<name>.py` with a class inheriting `TopologyMixin`
2. Implement `build_topology()` and `get_neighbourhood_best(i)`
3. Export from `topologies/__init__.py`
4. Register in `_TOPOLOGY_MAP` in `pso/factory.py`

## Adding a new operator variant

1. Create `pso/operators/<name>.py` with a class inheriting `DefaultOperatorsMixin` (or directly from `object` if a full replacement)
2. Override `crossover()` and/or `mutate()`
3. Export from `pso/operators/__init__.py`
4. Register in `_OPERATOR_MAP` in `pso/factory.py`

## Numba usage guidelines

- Distance matrix computation in `data/tsplib/reader.py` is already `@numba.njit`
- `PSOBase.path_length()` is the next candidate — swap in a `@njit` kernel when ready
- Crossover/mutate will require converting `self.particles` from `list[list[int]]` to `np.ndarray(shape=(N, n_city), dtype=int64)` — migration is isolated to `PSOBase`
- Avoid Python objects inside `@njit` functions — use plain numpy arrays
- Pre-compile with a small warm-up call when benchmarking
