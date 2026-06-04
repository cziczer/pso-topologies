# PSO-TSP

PhD research: application of **Particle Swarm Optimization (PSO)** to the **Travelling Salesman Problem (TSP)**, comparing the effect of different swarm topologies and velocity-update operators on solution quality and convergence behaviour.

- Python 3.13, managed with **uv**
- Numba JIT-compiled hot paths for performance
- Composable topology × operator variants via mixin architecture

## Package structure

```
pso/
  base.py              # PSOBase — core algorithm, main loop
  factory.py           # AlgorithmFactory — builds variants by name
  operators/
    default.py         # Order crossover (OX) + swap mutation
    cognitive.py       # Cognitive variant — adds personal-best crossover step
    apv.py             # APV — Adaptive Probability Vector (Numba-accelerated)
    swu.py             # SWU — Similarity-Weighted Update (extends APV)
    _numba_kernels.py  # Shared JIT kernels for APV/SWU
topologies/
  global_.py           # Fully connected (no topology)
  ring.py              # Ring
  tree.py              # Binary tree
  mesh.py              # 2D mesh grid
  torus.py             # Wrap-around torus grid
  free_scale.py        # Barabási-Albert scale-free network
  dynamic_similarity.py # Similarity-based dynamic topology (DynSim / DynOpp / DynMix)
data/
  tsplib/              # TSPLibReader + 47 TSPLIB benchmark instances
experiments/
  runner.py            # ExperimentRunner — N independent runs
  serialiser.py        # JSON save/load with aggregate statistics
main.py                # CLI entry point
```

## Setup

```bash
# Install all dependencies (creates .venv automatically)
uv sync
```

## Running experiments

### CLI

```bash
uv run main.py --algorithm <NAME> --problem <PROBLEM> [options]
```

**Required arguments:**

| Flag | Description |
|---|---|
| `-a`, `--algorithm` | Algorithm name — see formats below |
| `-p`, `--problem` | Problem name, format `tsplib-<instance>` e.g. `tsplib-berlin52` |

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `-n`, `--repetitions` | `30` | Number of independent runs |
| `--particles` | `700` | Swarm size |
| `--max-iter` | `20000` | Max iterations per run |
| `--neighbours-radius` | `2` | Topology neighbourhood radius (static topologies) |
| `--stagnation-window` | `100` | Early stop if last N recorded values unchanged; `0` disables |
| `-o`, `--output` | auto | Output JSON path |
| `--param KEY=VALUE` | — | Extra topology/operator parameter (repeatable) |
| `-q`, `--quiet` | — | Suppress per-run progress |

**Algorithm name format:** `PSO[-OperatorVariant][-Topology]`

Token matching is case-insensitive. Operator and topology token namespaces are disjoint — order does not matter.

### Operator variants

| Token | Description |
|---|---|
| *(omitted)* | **Default**: OX crossover toward neighbourhood best + swap mutation |
| `Cognitive` | Adds a personal-best (cognitive) crossover step before the social crossover, mirroring the full PSO velocity equation (inertia + cognitive + social) |
| `APV` | **Adaptive Probability Vector** — probability-based transposition update. Velocity is a probability vector over all (j,k) transpositions; updated each iteration from pbest/gbest transposition patterns. Fully Numba-accelerated. First run triggers compilation (~10 s); subsequent runs use the on-disk cache. |
| `SWU` | **Similarity-Weighted Update** — extends APV by scaling pbest/gbest components by the Hamming similarity between the particle and each guide. Focuses probability mass on the remaining differences when two tours are already close. |

APV and SWU accept extra hyperparameters via `--param`:

| Parameter | Default | Description |
|---|---|---|
| `omega` | `0.5` | Inertia weight |
| `c_p` | `0.5` | Personal (cognitive) acceleration |
| `c_g` | `0.5` | Social (global) acceleration |

### Topologies

| Token | Description |
|---|---|
| *(omitted)* | **Global** — all particles share one best |
| `Ring` | Circular list; `--neighbours-radius` hops each direction |
| `Tree` | Binary tree |
| `Mesh` | 2D grid |
| `Torus` | Wrap-around 2D grid |
| `FreeScale` | Barabási-Albert scale-free network; extra param: `ba_attachment` |
| `DynSim` | **Dynamic exploit** — neighbourhoods rebuilt every N iterations from pairwise path similarity; each particle's neighbours are its most *similar* peers (smallest edit distance) |
| `DynOpp` | **Dynamic explore** — same mechanism, but neighbours are the most *dissimilar* peers (largest edit distance); injects diversity |
| `DynMix` | **Dynamic mix** — alternates exploit and explore recalculation cycles (2 exploit → 1 explore by default) |

Dynamic topology parameters (pass via `--param`):

| Parameter | Default | Description |
|---|---|---|
| `recalc_interval` | `10` | Rebuild neighbourhoods every N iterations |
| `neighbor_pct` | `0.05` | Fraction of swarm per neighbourhood (5 % of 700 = 35 particles) |
| `explore_every` | `3` | `DynMix` only: explore fires every Nth recalculation |

Similarity is measured via normalised Hamming distance on paths rolled to start with city 0, so greedy-init starting cities don't inflate the distance.

### Algorithm name examples

| Name | Operators | Topology |
|---|---|---|
| `PSO` | Default | Global |
| `PSO-Ring` | Default | Ring |
| `PSO-Cognitive-Ring` | Cognitive | Ring |
| `PSO-APV` | APV | Global |
| `PSO-APV-Ring` | APV | Ring |
| `PSO-SWU-Torus` | SWU | Torus |
| `PSO-DynSim` | Default | Dynamic exploit |
| `PSO-DynOpp` | Default | Dynamic explore |
| `PSO-DynMix` | Default | Dynamic mix |
| `PSO-APV-DynSim` | APV | Dynamic exploit |

### CLI examples

```bash
# Ring topology, 30 runs, default settings
uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 30

# APV operators with ring topology
uv run main.py -a PSO-APV-Ring -p tsplib-berlin52 -n 30

# SWU operators, custom inertia and acceleration
uv run main.py -a PSO-SWU -p tsplib-berlin52 -n 10 --param omega=0.7 --param c_p=0.3 --param c_g=0.7

# Dynamic exploit topology — rebuild every 5 iterations, 10 % neighbours
uv run main.py -a PSO-DynSim -p tsplib-berlin52 -n 10 --param recalc_interval=5 --param neighbor_pct=0.1

# Dynamic explore topology
uv run main.py -a PSO-DynOpp -p tsplib-berlin52 -n 10

# Dynamic mix — explore fires every 2nd recalculation (1 exploit, 1 explore)
uv run main.py -a PSO-DynMix -p tsplib-berlin52 -n 10 --param explore_every=2

# APV with dynamic similarity topology
uv run main.py -a PSO-APV-DynSim -p tsplib-berlin52 -n 10

# No early stopping — always run to max_iter
uv run main.py -a PSO-Torus -p tsplib-berlin52 -n 10 --stagnation-window 0

# Scale-free with custom attachment parameter
uv run main.py -a PSO-FreeScale -p tsplib-berlin52 -n 10 --param ba_attachment=5

# Save to specific output file
uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 30 -o results/ring_berlin52.json
```

### Python API (Jupyter / scripts)

The full stack is importable directly — no CLI required:

```python
from data import TSPLibReader
from experiments import ExperimentRunner, ResultSerialiser

instance = TSPLibReader().load("data/tsplib/berlin52.tsp")

# Run an experiment
runner = ExperimentRunner(
    algorithm_name="PSO-DynMix",
    instance=instance,
    problem_name="tsplib-berlin52",
    n_repetitions=30,
    algorithm_kwargs={
        "num_particles": 700,
        "max_iter": 20_000,
        "recalc_interval": 10,
        "neighbor_pct": 0.05,
        "explore_every": 3,
    },
)
result = runner.run()
ResultSerialiser.save(result, "results/dynmix_berlin52.json")
```

Build and run a single PSO directly:

```python
from pso import AlgorithmFactory

# Dynamic exploit topology with custom parameters
pso = AlgorithmFactory.build(
    "PSO-DynSim",
    instance,
    num_particles=700,
    max_iter=20_000,
    recalc_interval=5,
    neighbor_pct=0.10,
)
result = pso.run()
print(result.best_path_length)

# APV operators + dynamic similarity topology
pso = AlgorithmFactory.build(
    "PSO-APV-DynSim",
    instance,
    num_particles=700,
    max_iter=20_000,
    omega=0.5, c_p=0.5, c_g=0.5,   # APV params
    recalc_interval=10,              # topology params
    neighbor_pct=0.05,
)
result = pso.run()
```

### Result JSON format

Each experiment produces one JSON file containing all N runs:

```json
{
  "schema_version": "1.0",
  "timestamp_utc": "2026-03-29T20:00:00Z",
  "problem": { "name": "tsplib-berlin52", "dimension": 52, "optimal_known": null },
  "algorithm": {
    "name": "PSO-DynMix",
    "topology": "dynmix",
    "config": { "num_particles": 700, "max_iter": 20000, "recalc_interval": 10 }
  },
  "aggregate": {
    "n_runs": 30,
    "best_path_length_mean": 8120.45,
    "best_path_length_std": 213.78,
    "best_path_length_min": 7891.23,
    "best_path_length_max": 8601.10,
    "best_path_length_median": 8094.67,
    "total_wall_time_seconds": 4823.1
  },
  "runs": [
    {
      "run_index": 0,
      "best_path_length": 7891.23,
      "best_path": [0, 12, 7, 43, "..."],
      "iteration_history": [9201.4, 8800.1, "..."],
      "iterations_run": 3420,
      "wall_time_seconds": 159.4
    }
  ]
}
```

`iteration_history` records the best length every 20 iterations — use it for convergence curve plots across runs.

## Available TSPLIB instances

47 benchmark instances are included in `data/tsplib/`. A selection:

| Instance | Cities | Notes |
|---|---|---|
| `berlin52` | 52 | Classic small benchmark |
| `gr96` | 96 | |
| `a280` | 280 | |
| `ali535` | 535 | |
| `rat575` | 575 | |
| `u724` | 724 | |
| `vm1084` | 1084 | |

## Development

```bash
# Add a new dependency
uv add <package>

# Jupyter
uv run jupyter notebook
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

## Changelog

### Dynamic similarity topologies (`DynSim` / `DynOpp` / `DynMix`)

Three new topology variants in `topologies/dynamic_similarity.py` that replace a fixed graph structure with neighbourhoods computed on-the-fly from pairwise path similarity.

Every `recalc_interval` iterations the swarm's current paths are compared using normalised Hamming distance (paths are rolled to start with city 0 before comparison). Each particle is then assigned the `neighbor_pct` fraction of the swarm as its neighbourhood — either the most similar peers (`DynSim`), the most dissimilar peers (`DynOpp`), or an alternating sequence of the two (`DynMix`).

The iteration boundary is detected inside `get_neighbourhood_best` when `particle_idx == 0`, requiring no changes to `PSOBase`.

### APV and SWU operators

Two new Numba-accelerated operator variants based on probability vectors over transpositions (position-swap pairs), fully described in Mastalerczyk et al., *"New update operators for discrete problems in Particle Swarm Optimisation"*, JOCS 2026.

**APV** (`pso/operators/apv.py`): each particle maintains a velocity vector whose components are probabilities over all (j,k) transposition pairs. Each iteration the velocity is updated from the particle's personal best and the neighbourhood best using the standard PSO inertia + cognitive + social formula, then each transposition is applied stochastically to generate a new position. The inner loop is JIT-compiled with `@njit(parallel=True)`.

**SWU** (`pso/operators/swu.py`): extends APV by multiplying the pbest and gbest components of the velocity update by the Hamming similarity between the current particle and each guide. When a particle is already close to its guide, the probability mass concentrates on the few remaining differences rather than being spread uniformly.

Both variants override `run()` and do not use `crossover()`/`mutate()`. They are composable with all topologies.

### Cognitive operator

`CognitiveMixin` (`pso/operators/cognitive.py`) wraps the default OX + swap operators with an additional personal-best crossover step before the social crossover, mirroring the full PSO velocity equation (inertia + cognitive + social component).
