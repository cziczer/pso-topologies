# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PhD research: application of **PSO (Particle Swarm Optimization)** to **TSP (Travelling Salesman Problem)**.

- Python 3.13, managed with **uv** (`pyproject.toml`)
- **Numba** used for JIT-compiled hot paths (distance matrix already JIT-compiled; APV/SWU operators fully JIT)
- Old reference code is the **primary source** for porting logic with improvements — path: `~/Downloads/old_proj/phd/_deprecated/PSO_old/`

## Package structure

```
pso/
  base.py              # PSOBase — swarm state, greedy init, main loop, topology/operator hooks
  factory.py           # AlgorithmFactory — parses name strings, assembles composed classes
  operators/
    default.py         # DefaultOperatorsMixin — order crossover (OX) + swap mutation
    cognitive.py       # CognitiveMixin — adds personal-best crossover step before social step
    apv.py             # APVOperatorsMixin — Adaptive Probability Vector; Numba-accelerated main loop
    swu.py             # SWUOperatorsMixin — Similarity-Weighted Update; extends APV with sim coefficients
    _numba_kernels.py  # @njit kernels shared by APV and SWU (apv_full_iteration, swu_full_iteration)
topologies/
  base.py              # TopologyMixin ABC — cooperative __init__, build_topology hook
  global_.py           # GlobalTopologyMixin — no topology, all particles share global best
  ring.py              # RingTopologyMixin
  tree.py              # TreeTopologyMixin
  mesh.py              # MeshTopologyMixin
  torus.py             # TorusTopologyMixin
  free_scale.py        # FreeScaleTopologyMixin — Barabási-Albert scale-free graph
  dynamic_similarity.py # DynamicSimilarityTopologyMixin + DynOpp + DynMix — similarity-based dynamic topology
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

MRO: `OperatorMixin → TopologyMixin → PSOBase`. The main loop lives in `PSOBase.run()` (or is overridden by `APVOperatorsMixin.run()` for APV/SWU) and calls two hooks:
- `get_neighbourhood_best(i)` — provided by topology mixin
- `crossover(current, guide)` / `mutate(path)` — provided by operator mixin (not used by APV/SWU)

**Algorithm name format**: `PSO[-OperatorVariant][-Topology]`
- `"PSO"` → global topology + default operators
- `"PSO-Ring"` → ring topology + default operators
- `"PSO-APV-Ring"` → ring topology + APV operators
- `"PSO-DynMix"` → dynamic mix topology + default operators

## Operator variants

| Token | Class | Description |
|---|---|---|
| *(omitted)* | `DefaultOperatorsMixin` | OX crossover toward neighbourhood best + swap mutation |
| `Cognitive` | `CognitiveMixin` | Adds personal-best (cognitive) crossover step before the social step |
| `APV` | `APVOperatorsMixin` | Adaptive Probability Vector — probability-based transposition update, Numba-parallel |
| `SWU` | `SWUOperatorsMixin` | Similarity-Weighted Update — APV scaled by particle-to-guide similarity |

APV and SWU override `run()` entirely; they do not use `crossover()`/`mutate()`.

## Topologies

| Token | Class | Description |
|---|---|---|
| *(omitted)* | `GlobalTopologyMixin` | All particles share one global best |
| `Ring` | `RingTopologyMixin` | Circular list; `neighbours_radius` hops each direction |
| `Tree` | `TreeTopologyMixin` | Binary tree |
| `Mesh` | `MeshTopologyMixin` | 2D grid |
| `Torus` | `TorusTopologyMixin` | Wrap-around 2D grid |
| `FreeScale` | `FreeScaleTopologyMixin` | Barabási-Albert scale-free graph |
| `DynSim` | `DynamicSimilarityTopologyMixin` | Dynamic: neighbours = most *similar* paths (exploit) |
| `DynOpp` | `DynamicOppositeTopologyMixin` | Dynamic: neighbours = most *dissimilar* paths (explore) |
| `DynMix` | `DynamicMixTopologyMixin` | Dynamic: alternates exploit/explore each recalculation cycle |

### Dynamic similarity topology parameters

Similarity is measured via normalised Hamming distance on canonical paths (rolled to start with city 0).
Neighbourhoods are rebuilt every `recalc_interval` iterations; iteration boundary is detected when
`get_neighbourhood_best` is called with `particle_idx == 0`.

| Parameter | Default | Description |
|---|---|---|
| `recalc_interval` | `10` | Rebuild neighbourhoods every N iterations |
| `neighbor_pct` | `0.05` | Fraction of swarm per neighbourhood (5 % of 700 = 35 particles) |
| `explore_every` | `3` | `DynMix` only: explore fires every Nth recalculation; others are exploit |

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
2. Override `crossover()` and/or `mutate()` (or `run()` for full loop control like APV/SWU)
3. Export from `pso/operators/__init__.py`
4. Register in `_OPERATOR_MAP` in `pso/factory.py`

## Numba usage guidelines

- Distance matrix computation in `data/tsplib/reader.py` is `@numba.njit`
- APV/SWU inner loops (`_numba_kernels.py`) are `@njit(parallel=True)` — first call triggers compilation (a few seconds); subsequent runs use the on-disk cache in `__pycache__`
- `PSOBase.path_length()` is the next candidate for JIT — swap in a `@njit` kernel when ready
- Crossover/mutate will require converting `self.particles` from `list[list[int]]` to `np.ndarray(shape=(N, n_city), dtype=int64)` — migration is isolated to `PSOBase`
- Avoid Python objects inside `@njit` functions — use plain numpy arrays
- Pre-compile with a small warm-up call when benchmarking
