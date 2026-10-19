#!/usr/bin/env python3
"""Work out which experiment configs still need running, and how to pack them.

Why this exists
---------------
The first three SLURM submissions each sliced the plan into fixed-size index
blocks (50, then 25 configs per array task) and reserved 16 CPUs per task for the
whole walltime.  Per-config cost spans three orders of magnitude, so every task
ended up holding all 16 CPUs while one straggler finished -- or was killed at the
wall with that straggler incomplete.  Measured outcome: ~26 583 core-h billed for
~9 212 core-h of actual experiment work (35 % efficiency), and 15 of 38 array
tasks died at the wall.  That is what exhausted the first 25 000 h allocation.

This script fixes both halves of that:

1. It derives the *pending* set fresh from results/logs/ instead of trusting a
   snapshot, so re-running it after a partial job never re-runs completed work.
   Completion is keyed on the exact command string plus ``exit_code: 0``, which
   is index-independent -- renumbering the plan cannot corrupt it.
2. It greedily bin-packs the pending configs into equal-*cost* array tasks using
   per-(problem, family) medians measured from the completed runs, and emits the
   contiguous ``--start/--end`` ranges for those bins.  Equal cost per task is
   what keeps reserved cores busy.

Usage
-----
    uv run experiments/find_missing.py                       # report only
    uv run experiments/find_missing.py --tasks 12 \
        --out results/pending.log --split-out results/pending.split

The ``--out`` file is in the same JSON-lines retry format that
``run_parallel.py --retry-failed`` consumes.  It is written in bin order, so
array task *i* runs exactly the slice named in the ``--split-out`` table.
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

# Dynamic-topology group label in the active plan (was H-M in the retired one).
DYN_GROUPS = {"DYN"}


def _load_runner():
    """Import run_parallel.py as a module without executing its main()."""
    path = Path(__file__).with_name("run_parallel.py")
    spec = importlib.util.spec_from_file_location("_run_parallel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_parallel"] = mod        # dataclasses needs it importable
    spec.loader.exec_module(mod)
    return mod


def family(algo: str) -> str:
    """Coarse cost class.  APV/SWU override run() and are far more expensive."""
    for key in ("APV", "SWU", "Cognitive-Explore", "Explore", "Cognitive"):
        if key in algo:
            return key
    return "Default"


def completed_costs() -> dict[str, float]:
    """Map successful command string -> elapsed hours, from results/logs/*.log."""
    done: dict[str, float] = {}
    for path in glob.glob("results/logs/*.log"):
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        cmd = re.search(r"^CMD: (.+)$", text, re.M)
        code = re.search(r"^exit_code: (-?\d+)$", text, re.M)
        secs = re.search(r"^elapsed_s: ([\d.]+)$", text, re.M)
        if cmd and code and code.group(1) == "0":
            done[cmd.group(1).strip()] = float(secs.group(1)) / 3600 if secs else 0.0
    return done


def build_cost_model(rp, plan, done: dict[str, float]):
    """Median hours per (problem, family), with per-problem and global fallbacks."""
    buckets: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    per_problem: dict[str, list[float]] = collections.defaultdict(list)
    for exp in plan:
        cmd = " ".join(exp.to_cmd())
        if cmd in done and done[cmd] > 0:
            buckets[(exp.problem, family(exp.algo))].append(done[cmd])
            per_problem[exp.problem].append(done[cmd])

    fine = {k: statistics.median(v) for k, v in buckets.items()}
    coarse = {k: statistics.median(v) for k, v in per_problem.items()}
    measured = [v for v in done.values() if v > 0]
    overall = statistics.median(measured) if measured else 1.0

    def estimate(exp) -> float:
        # Dynamic topologies have no completed runs at all, so they fall through
        # to the per-problem median.  Treat that as a floor, not a measurement --
        # see --dyn-factor.
        base = (fine.get((exp.problem, family(exp.algo)))
                or coarse.get(exp.problem)
                or overall)
        # Every measurement above comes from a 10-repetition run.  Cost is
        # essentially linear in repetitions (they run sequentially in one
        # process), so rescale when this config uses a different count --
        # otherwise large instances, now at 5 repetitions, are estimated 2x high.
        return base * exp.repetitions() / rp.REPS_DEFAULT

    return estimate


def pack(items, n_bins: int):
    """Greedy longest-processing-time bin packing.  Returns a list of bins."""
    bins: list[list] = [[] for _ in range(n_bins)]
    loads = [0.0] * n_bins
    for entry in sorted(items, key=lambda t: -t[1]):
        i = loads.index(min(loads))
        bins[i].append(entry)
        loads[i] += entry[1]
    return bins


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report and pack the experiment configs that still need running",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-dim", type=int, default=None, metavar="D",
                    help="Only consider problems with dimension <= D "
                         "(default: the runner's active scope)")
    ap.add_argument("--all-problems", action="store_true",
                    help="Consider every instance in ALL_PROBLEMS")
    ap.add_argument("--problems", nargs="+", metavar="P",
                    help="Restrict to these instances, e.g. --problems ali535 gr666")
    ap.add_argument("--groups", nargs="+", metavar="G",
                    help="Restrict to specific groups, e.g. --groups H I J")
    ap.add_argument("--tasks", type=int, default=12, metavar="N",
                    help="Number of array tasks to bin-pack into (default: 12)")
    ap.add_argument("--dyn-factor", type=float, default=1.0, metavar="F",
                    help="Multiply estimated cost of dynamic-topology configs "
                         "(groups H-M) by F.  They have no measured cost yet, so "
                         "F>1 buys margin when sizing walltime.")
    ap.add_argument("--out", metavar="FILE",
                    help="Write pending configs as a --retry-failed JSON-lines log")
    ap.add_argument("--split-out", metavar="FILE",
                    help="Write a bash-sourceable START/END table for the array")
    args = ap.parse_args()

    rp = _load_runner()

    if args.problems:
        unknown = [p for p in args.problems if p not in rp.ALL_PROBLEMS]
        if unknown:
            sys.exit(f"unknown problem(s): {', '.join(unknown)}")
        problems = list(args.problems)
    elif args.all_problems:
        problems = list(rp.ALL_PROBLEMS)
    elif args.max_dim is not None:
        problems = [p for p in rp.ALL_PROBLEMS if rp.PROBLEM_DIMS[p] <= args.max_dim]
    else:
        problems = list(rp.PROBLEMS)

    plan_scoped = rp.build_experiments(problems=problems, groups=args.groups)
    plan_full = rp.build_legacy_experiments(problems=rp.ALL_PROBLEMS)
    done = completed_costs()
    estimate = build_cost_model(rp, plan_full, done)

    def cost_of(exp) -> float:
        c = estimate(exp)
        return c * args.dyn_factor if exp.group in DYN_GROUPS else c

    pending = [e for e in plan_scoped if " ".join(e.to_cmd()) not in done]

    print(f"Problem scope      : {', '.join(problems)}")
    print(f"Plan in scope      : {len(plan_scoped)} configs "
          f"({len(plan_scoped) * 10} runs)")
    print(f"Already completed  : {len(plan_scoped) - len(pending)} configs")
    print(f"Still pending      : {len(pending)} configs ({len(pending) * 10} runs)")

    if not pending:
        print("\nNothing pending in this scope.")
        return

    total = sum(cost_of(e) for e in pending)
    print(f"Estimated compute  : ~{total:,.0f} core-h (dyn-factor {args.dyn_factor:g})")

    by_problem = collections.Counter(e.problem for e in pending)
    print("\nPending by problem:")
    for p in problems:
        if by_problem[p]:
            sub = sum(cost_of(e) for e in pending if e.problem == p)
            print(f"  {p:<10} {by_problem[p]:>4} configs  ~{sub:>7,.0f} core-h")

    n_dyn = sum(1 for e in pending if e.group in DYN_GROUPS)
    if n_dyn:
        pct = 100 * n_dyn / len(pending)
        print(f"\nNote: {n_dyn} of {len(pending)} pending configs ({pct:.0f} %) are "
              f"dynamic topologies")
        print("      (groups H-M) with NO completed runs anywhere, so their cost is")
        print("      extrapolated from static-topology runs on the same problem.")
        print("      Re-run this script once a few have finished for a measured")
        print("      estimate, or pass --dyn-factor 2 to size walltime defensively.")

    items = [(e, cost_of(e)) for e in pending]
    n_bins = max(1, min(args.tasks, len(items)))
    bins = pack(items, n_bins)

    print(f"\nCost-balanced split into {n_bins} array tasks:")
    print(f"  {'task':>4}{'configs':>9}{'est core-h':>12}  slice")
    ordered: list = []
    ranges: list[tuple[int, int]] = []
    for i, b in enumerate(bins):
        start = len(ordered) + 1
        ordered.extend(e for e, _ in b)
        end = len(ordered)
        ranges.append((start, end))
        print(f"  {i:>4}{len(b):>9}{sum(c for _, c in b):>12,.1f}"
              f"  --start {start} --end {end}")

    loads = [sum(c for _, c in b) for b in bins]
    print(f"\n  heaviest task {max(loads):,.1f} core-h, lightest {min(loads):,.1f} "
          f"-- spread {max(loads) - min(loads):,.1f} (lower is better packing)")
    slowest = max(c for _, c in items)
    print(f"  slowest single config estimate: {slowest:,.1f} h "
          f"-- array walltime MUST exceed this")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for seq, exp in enumerate(ordered, 1):
                fh.write(json.dumps({
                    "seq": seq,
                    "label": f"[{exp.group}] {exp.algo} @ {exp.problem} "
                             f"{' '.join(exp.extra)}".strip(),
                    "command": " ".join(exp.to_cmd()),
                    "_retry": {
                        "group": exp.group,
                        "algo": exp.algo,
                        "problem": exp.problem,
                        "extra": exp.extra,
                        "reps": exp.repetitions(),
                    },
                }) + "\n")
        print(f"\nWrote {len(ordered)} pending configs -> {out}")

    if args.split_out:
        sp = Path(args.split_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("w") as fh:
            fh.write("# Generated by experiments/find_missing.py -- do not edit.\n")
            fh.write("# Cost-balanced array slices into the pending log.\n")
            fh.write(f"N_TASKS={len(ranges)}\n")
            fh.write("TASK_START=(%s)\n" % " ".join(str(a) for a, _ in ranges))
            fh.write("TASK_END=(%s)\n" % " ".join(str(b) for _, b in ranges))
        print(f"Wrote array split table       -> {sp}")
        print(f"Set the array directive to:     #SBATCH --array=0-{len(ranges) - 1}")


if __name__ == "__main__":
    main()
