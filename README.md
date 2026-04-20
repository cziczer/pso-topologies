# PSO-TSP

PhD research: application of **Particle Swarm Optimization (PSO)** to the **Travelling Salesman Problem (TSP)**, comparing the effect of different swarm topologies on solution quality and convergence behaviour.

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
topologies/
  global_.py           # Fully connected (no topology)
  ring.py              # Ring
  tree.py              # Binary tree
  mesh.py              # 2D mesh grid
  torus.py             # Wrap-around torus grid
  free_scale.py        # Barabási-Albert scale-free network
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
| `--neighbours-radius` | `2` | Topology neighbourhood radius |
| `--stagnation-window` | `100` | Early stop if last N recorded values unchanged; `0` disables |
| `-o`, `--output` | auto | Output JSON path |
| `--param KEY=VALUE` | — | Extra topology-specific parameter (repeatable) |
| `-q`, `--quiet` | — | Suppress per-run progress |

**Algorithm name format:** `PSO[-OperatorVariant][-Topology]`

| Name | Operators | Topology |
|---|---|---|
| `PSO` | Default (OX + swap) | Global (all particles share one best) |
| `PSO-Ring` | Default | Ring |
| `PSO-Tree` | Default | Binary tree |
| `PSO-Mesh` | Default | 2D mesh grid |
| `PSO-Torus` | Default | Wrap-around torus |
| `PSO-FreeScale` | Default | Barabási-Albert scale-free |
| `PSO-Cognitive` | Cognitive (OX + swap + personal best) | Global |
| `PSO-Cognitive-Ring` | Cognitive | Ring |
| `PSO-Cognitive-<Topology>` | Cognitive | Any topology above |

**Operator variants:**

| Variant token | Description |
|---|---|
| *(omitted)* | Default: OX crossover toward neighbourhood best + swap mutation |
| `Cognitive` | Adds a personal-best (cognitive) crossover step before the social crossover, mirroring the full PSO velocity equation (inertia + cognitive + social) |

**Examples:**

```bash
# Ring topology, 30 runs, default settings
uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 30

# Quick sanity check (few particles, short run)
uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 1 --particles 50 --max-iter 1000

# No early stopping — always run to max_iter
uv run main.py -a PSO-Torus -p tsplib-berlin52 -n 10 --stagnation-window 0

# Scale-free with custom attachment parameter
uv run main.py -a PSO-FreeScale -p tsplib-berlin52 -n 10 --param ba_attachment=5

# Torus with custom grid width
uv run main.py -a PSO-Torus -p tsplib-a280 -n 10 --param grid_width=60

# Save to specific output file
uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 30 -o results/ring_berlin52.json
```

### Python API (Jupyter / scripts)

The full stack is importable directly — no CLI required:

```python
from data import TSPLibReader
from experiments import ExperimentRunner, ResultSerialiser

# Load a TSP instance
instance = TSPLibReader().load("data/tsplib/berlin52.tsp")

# Run an experiment
runner = ExperimentRunner(
    algorithm_name="PSO-Ring",
    instance=instance,
    problem_name="tsplib-berlin52",
    n_repetitions=30,
    algorithm_kwargs={
        "num_particles": 700,
        "max_iter": 20_000,
        "neighbours_radius": 2,
        "stagnation_window": None,  # run to max_iter, no early stop
    },
)
result = runner.run()
ResultSerialiser.save(result, "results/ring_berlin52.json")
```

Build and run a single PSO directly (no experiment wrapper):

```python
from pso import AlgorithmFactory

pso = AlgorithmFactory.build("PSO-Ring", instance, num_particles=700, max_iter=20_000)
run_result = pso.run()
print(run_result.best_path_length)
```

### Result JSON format

Each experiment produces one JSON file containing all N runs:

```json
{
  "schema_version": "1.0",
  "timestamp_utc": "2026-03-29T20:00:00Z",
  "problem": { "name": "tsplib-berlin52", "dimension": 52, "optimal_known": null },
  "algorithm": {
    "name": "PSO-Ring",
    "topology": "ring",
    "config": { "num_particles": 700, "max_iter": 20000, "neighbours_radius": 2 }
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
