"""Entry point for PSO-TSP experiments.

Usage examples::

    uv run main.py --algorithm PSO-Ring --problem tsplib-berlin52 --repetitions 30
    uv run main.py -a PSO-Torus -p tsplib-a280 -n 5 --max-iter 5000 --stagnation-window 0
    uv run main.py -a PSO-FreeScale -p tsplib-berlin52 --param ba_attachment=7
    uv run main.py -a PSO -p tsplib-berlin52 --stagnation-window 0  # run to max_iter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Problem resolution
# ---------------------------------------------------------------------------

def resolve_problem(problem_name: str, data_root: str = "data") -> "TSPInstance":
    """Parse *problem_name* and load the corresponding TSP instance.

    Format: ``<source>-<instance>``  e.g. ``tsplib-berlin52``

    Raises:
        ValueError: for unknown source prefixes or missing instances.
    """
    from data import TSPLibReader

    parts = problem_name.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"Problem name must be '<source>-<instance>', got: '{problem_name}'\n"
            "Example: tsplib-berlin52"
        )
    source, instance = parts

    if source == "tsplib":
        path = Path(data_root) / "tsplib" / f"{instance}.tsp"
        if not path.exists():
            available = sorted(p.stem for p in (Path(data_root) / "tsplib").glob("*.tsp"))
            raise ValueError(
                f"Instance '{instance}' not found at '{path}'.\n"
                f"Available instances: {available}"
            )
        return TSPLibReader().load(str(path))

    raise ValueError(
        f"Unknown problem source: '{source}'.  Currently supported: 'tsplib'"
    )


# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------

def _default_output_path(problem_name: str, algorithm_name: str) -> Path:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / f"{problem_name}_{algorithm_name}_{ts}.json"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="uv run main.py",
        description="Run PSO-TSP experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run main.py -a PSO-Ring -p tsplib-berlin52 -n 30\n"
            "  uv run main.py -a PSO-Torus -p tsplib-a280 -n 10 --max-iter 5000\n"
            "  uv run main.py -a PSO-FreeScale -p tsplib-berlin52 --param ba_attachment=7\n"
            "  uv run main.py -a PSO -p tsplib-berlin52 --stagnation-window 0\n"
        ),
    )

    p.add_argument(
        "--algorithm", "-a",
        required=True,
        metavar="NAME",
        help=(
            "Algorithm name. Format: PSO[-Topology]. "
            "Topologies: Ring, Tree, Mesh, Torus, FreeScale. "
            "Examples: PSO, PSO-Ring, PSO-Torus"
        ),
    )
    p.add_argument(
        "--problem", "-p",
        required=True,
        metavar="NAME",
        help="Problem name. Format: tsplib-<instance>. Example: tsplib-berlin52",
    )
    p.add_argument(
        "--repetitions", "-n",
        type=int,
        default=30,
        metavar="N",
        help="Number of independent experiment repetitions (default: 30).",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Output JSON path. Default: results/<problem>_<algorithm>_<timestamp>.json",
    )
    p.add_argument(
        "--particles",
        type=int,
        default=700,
        help="Swarm size (default: 700).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=20_000,
        dest="max_iter",
        help="Maximum iterations per run (default: 20000).",
    )
    p.add_argument(
        "--neighbours-radius",
        type=int,
        default=2,
        dest="neighbours_radius",
        help="Neighbourhood radius for topology variants (default: 2).",
    )
    p.add_argument(
        "--stagnation-window",
        type=int,
        default=100,
        dest="stagnation_window",
        metavar="N",
        help=(
            "Early stopping: halt if the last N recorded values are unchanged "
            "(default: 100). Set to 0 to disable early stopping and always run "
            "to --max-iter."
        ),
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra algorithm parameter (repeatable). "
            "Values are auto-cast to int or float when possible. "
            "Example: --param ba_attachment=7 --param grid_width=60"
        ),
    )
    p.add_argument(
        "--data-root",
        default="data",
        metavar="DIR",
        help="Root directory for problem data files (default: data).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-run progress output.",
    )

    return p


def _parse_extra_params(params: list[str]) -> dict:
    """Convert ['key=value', ...] into a dict with auto-typed values."""
    result: dict = {}
    for item in params:
        if "=" not in item:
            raise ValueError(f"--param must be in KEY=VALUE format, got: '{item}'")
        key, raw_value = item.split("=", maxsplit=1)
        # Auto-cast: try int, then float, else keep as string
        try:
            result[key] = int(raw_value)
        except ValueError:
            try:
                result[key] = float(raw_value)
            except ValueError:
                result[key] = raw_value
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from experiments import ExperimentRunner, ResultSerialiser

    parser = build_parser()
    args = parser.parse_args()

    # Load instance
    try:
        instance = resolve_problem(args.problem, data_root=args.data_root)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build algorithm kwargs
    algorithm_kwargs: dict = {
        "num_particles": args.particles,
        "max_iter": args.max_iter,
        "neighbours_radius": args.neighbours_radius,
        "stagnation_window": args.stagnation_window if args.stagnation_window > 0 else None,
    }

    # Merge extra --param flags
    try:
        extra = _parse_extra_params(args.param)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    algorithm_kwargs.update(extra)

    # Determine output path
    output_path = Path(args.output) if args.output else _default_output_path(
        args.problem, args.algorithm
    )

    print(f"Algorithm : {args.algorithm}")
    print(f"Problem   : {args.problem}  (dimension={instance.dimension})")
    print(f"Runs      : {args.repetitions}")
    print(f"Output    : {output_path}")
    print()

    # Run experiment
    try:
        runner = ExperimentRunner(
            algorithm_name=args.algorithm,
            instance=instance,
            problem_name=args.problem,
            n_repetitions=args.repetitions,
            algorithm_kwargs=algorithm_kwargs,
            verbose=not args.quiet,
        )
        result = runner.run()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Save results
    ResultSerialiser.save(result, output_path)

    agg = {
        "mean":   sum(r.best_path_length for r in result.runs) / len(result.runs),
        "best":   min(r.best_path_length for r in result.runs),
        "worst":  max(r.best_path_length for r in result.runs),
    }
    print(f"\nDone. Results saved to: {output_path}")
    print(f"  best={agg['best']:.2f}  mean={agg['mean']:.2f}  worst={agg['worst']:.2f}  "
          f"total_time={result.total_wall_time_seconds:.1f}s")


if __name__ == "__main__":
    main()
