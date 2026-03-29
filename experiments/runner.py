from __future__ import annotations

import time
from datetime import datetime, timezone

from data.loader import TSPInstance
from experiments.result import ExperimentResult, RunResult
from pso.factory import AlgorithmFactory


class ExperimentRunner:
    """Run *n_repetitions* independent PSO runs and collect results.

    This class is the primary entry point for both CLI and Python/Jupyter usage::

        runner = ExperimentRunner(
            algorithm_name="PSO-Ring",
            instance=instance,
            problem_name="tsplib-berlin52",
            n_repetitions=30,
            algorithm_kwargs={"num_particles": 700, "max_iter": 20_000},
        )
        result = runner.run()

    Args:
        algorithm_name: Algorithm name string passed to :class:`AlgorithmFactory`.
        instance: Loaded TSP problem instance.
        problem_name: Human-readable problem identifier (e.g. ``"tsplib-berlin52"``).
        n_repetitions: Number of independent runs.
        algorithm_kwargs: Dict of keyword arguments forwarded to the algorithm
            constructor.  All keys/values are stored verbatim in the result JSON.
        known_optimal: Known optimal tour length for the instance, or ``None``.
        verbose: Print progress to stdout (default ``True``).
    """

    def __init__(
        self,
        algorithm_name: str,
        instance: TSPInstance,
        problem_name: str,
        n_repetitions: int,
        algorithm_kwargs: dict | None = None,
        known_optimal: float | None = None,
        verbose: bool = True,
    ) -> None:
        self.algorithm_name = algorithm_name
        self.instance = instance
        self.problem_name = problem_name
        self.n_repetitions = n_repetitions
        self.algorithm_kwargs: dict = algorithm_kwargs or {}
        self.known_optimal = known_optimal
        self.verbose = verbose

    def run(self) -> ExperimentResult:
        """Execute all repetitions and return an :class:`ExperimentResult`."""
        runs: list[RunResult] = []
        total_start = time.perf_counter()

        for rep in range(self.n_repetitions):
            if self.verbose:
                print(f"  Run {rep + 1}/{self.n_repetitions} ...", end=" ", flush=True)

            pso = AlgorithmFactory.build(
                self.algorithm_name,
                self.instance,
                **self.algorithm_kwargs,
            )

            run_start = time.perf_counter()
            result = pso.run()
            wall_time = time.perf_counter() - run_start

            result.run_index = rep
            result.wall_time_seconds = wall_time
            runs.append(result)

            if self.verbose:
                print(
                    f"best={result.best_path_length:.2f}  "
                    f"iters={result.iterations_run}  "
                    f"time={wall_time:.1f}s"
                )

        total_time = time.perf_counter() - total_start

        return ExperimentResult(
            algorithm_name=self.algorithm_name,
            problem_name=self.problem_name,
            problem_dimension=self.instance.dimension,
            problem_optimal=self.known_optimal,
            algorithm_config=dict(self.algorithm_kwargs),
            runs=runs,
            total_wall_time_seconds=total_time,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
