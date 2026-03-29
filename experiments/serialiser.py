from __future__ import annotations

import json
import statistics
from pathlib import Path

from experiments.result import ExperimentResult, RunResult


class ResultSerialiser:
    """Serialise and deserialise :class:`ExperimentResult` to/from JSON."""

    @staticmethod
    def to_dict(result: ExperimentResult) -> dict:
        lengths = [r.best_path_length for r in result.runs]

        # Parse algorithm name into components for the JSON
        try:
            from pso.factory import AlgorithmFactory
            _, op_key, topo_key = AlgorithmFactory.parse(result.algorithm_name)
        except Exception:
            op_key, topo_key = "unknown", "unknown"

        # Split problem name into source + instance
        parts = result.problem_name.split("-", maxsplit=1)
        source = parts[0] if len(parts) == 2 else result.problem_name
        instance_name = parts[1] if len(parts) == 2 else result.problem_name

        return {
            "schema_version": "1.0",
            "timestamp_utc": result.timestamp_utc,
            "problem": {
                "name": result.problem_name,
                "source": source,
                "instance_name": instance_name,
                "dimension": result.problem_dimension,
                "optimal_known": result.problem_optimal,
            },
            "algorithm": {
                "name": result.algorithm_name,
                "base": "PSO",
                "operator_variant": op_key,
                "topology": topo_key,
                "config": result.algorithm_config,
            },
            "aggregate": {
                "n_runs": len(result.runs),
                "best_path_length_mean": statistics.mean(lengths),
                "best_path_length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
                "best_path_length_min": min(lengths),
                "best_path_length_max": max(lengths),
                "best_path_length_median": statistics.median(lengths),
                "total_wall_time_seconds": result.total_wall_time_seconds,
            },
            "runs": [
                {
                    "run_index": r.run_index,
                    "best_path_length": r.best_path_length,
                    "best_path": r.best_path,
                    "iteration_history": r.iteration_history,
                    "iterations_run": r.iterations_run,
                    "wall_time_seconds": r.wall_time_seconds,
                }
                for r in result.runs
            ],
        }

    @staticmethod
    def save(result: ExperimentResult, path: str | Path) -> None:
        """Write *result* as JSON to *path*, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(ResultSerialiser.to_dict(result), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> ExperimentResult:
        """Load an :class:`ExperimentResult` from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        runs = [
            RunResult(
                run_index=r["run_index"],
                best_path_length=r["best_path_length"],
                best_path=r["best_path"],
                iteration_history=r["iteration_history"],
                iterations_run=r["iterations_run"],
                wall_time_seconds=r["wall_time_seconds"],
            )
            for r in data["runs"]
        ]

        return ExperimentResult(
            algorithm_name=data["algorithm"]["name"],
            problem_name=data["problem"]["name"],
            problem_dimension=data["problem"]["dimension"],
            problem_optimal=data["problem"]["optimal_known"],
            algorithm_config=data["algorithm"]["config"],
            runs=runs,
            total_wall_time_seconds=data["aggregate"]["total_wall_time_seconds"],
            timestamp_utc=data["timestamp_utc"],
        )
