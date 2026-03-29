from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    """Result from a single PSO run."""

    run_index: int
    best_path_length: float
    best_path: list[int]
    iteration_history: list[float]  # best length recorded every 20 iterations
    iterations_run: int
    wall_time_seconds: float


@dataclass
class ExperimentResult:
    """Aggregated result from N independent PSO runs on one problem."""

    # Identity
    algorithm_name: str
    problem_name: str
    problem_dimension: int
    problem_optimal: float | None

    # Full config snapshot (all kwargs passed to AlgorithmFactory.build)
    algorithm_config: dict

    # Per-run data
    runs: list[RunResult]

    # Timing
    total_wall_time_seconds: float
    timestamp_utc: str
